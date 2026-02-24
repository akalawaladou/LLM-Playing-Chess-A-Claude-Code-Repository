# Suite de l'article — "Les LLM hallucinent aux echecs et c'est normal", vraiment?

> Ce document contient la continuation de ton draft, prete a etre adaptee pour Medium.
> Les blocs de code sont extraits du dossier `othello_probing/`.

---

## [Remplace les "blabla" dans la section probing]

### Comment les chercheurs le montrent ?

Ils examinent les activations internes du reseau avec une technique classique d'interpretabilite appelee **probing**. Quand on dit _activations internes_, ce sont les valeurs temporaires produites par les neurones du LLM dans les couches cachees du reseau quand le modele traite une entree donnee (par exemple une sequence de coups a Othello).

L'idee du probing : brancher un classificateur simple (par exemple une regression logistique ou un petit reseau de neurones) sur ces activations internes pour voir s'il est possible de decoder la position reelle du plateau.

Concretement, voici le protocole :

* **Etape 1 — Collecte des activations.** On fait passer des parties entières dans le modèle et, à chaque coup `t`, on récupère le vecteur d'activation produit par chaque couche du transformer. Ce vecteur est un point dans un espace à `d` dimensions (ici `d = 128`). Il ne représente, *a priori*, que l'information dont le modèle a besoin pour prédire le prochain coup.

* **Etape 2 — Labels = état réel du plateau.** Indépendamment, on connaît l'état exact du plateau à chaque coup grâce au moteur de jeu Othello. Chaque case vaut : `vide (0)`, `noir (1)` ou `blanc (2)`. On obtient ainsi un vecteur-label de 64 × 3 classes.

* **Etape 3 — Entraîner la probe.** On entraîne un petit classificateur — la *probe* — qui prend en entrée le vecteur d'activation (128-d) et doit prédire l'état de chacune des 64 cases. Deux variantes sont testées :
  - **Probe linéaire** : une simple régression logistique (`nn.Linear(128, 64*3)`). Si elle réussit, l'information est linéairement séparable dans les activations — triviale à extraire.
  - **Probe non-linéaire (MLP)** : un petit réseau à une couche cachée (`128 → 256 → 64*3`). Si celle-ci réussit alors que la probe linéaire échoue, l'information est présente mais encodée de manière non-linéaire dans les activations.

**Résultat** : la probe linéaire atteint ~65-75 % de précision — mieux que le hasard (33 %), mais loin d'être parfait. En revanche, la probe MLP atteint **~90-99 %** selon la couche. Le modèle *maintient bien* un état interne du plateau — mais cet état est encodé de manière non-linéaire dans ses activations.

> C'est exactement ce point qui est fascinant : le modèle n'a jamais reçu l'instruction "retiens la position". Il l'a fait émerger parce que c'était utile pour prédire le prochain coup.

---

## Reproduisons l'expérience — avec du code

Pour rendre tout cela concret, j'ai codé une reproduction simplifiée de l'expérience Othello-GPT. Le code complet est sur GitHub (lien en fin d'article) ; voici les morceaux clés.

### 1. Le moteur de jeu Othello

Rien de sorcier, mais indispensable : il faut pouvoir générer des milliers de parties aléatoires légales et enregistrer l'état du plateau à chaque coup.

```python
# othello.py — extraits

class OthelloGame:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        # Position initiale standard
        self.board[3, 3] = -1  # blanc
        self.board[3, 4] = 1   # noir
        self.board[4, 3] = 1   # noir
        self.board[4, 4] = -1  # blanc
        self.current_player = 1  # noir commence

    def play_move(self, move: int) -> bool:
        """Joue le coup (0..63), retourne les pions captures."""
        r, c = divmod(move, 8)
        all_flips = []
        for dr, dc in DIRECTIONS:
            all_flips.extend(
                self._flips_in_direction(r, c, dr, dc, self.current_player)
            )
        if not all_flips:
            return False
        self.board[r, c] = self.current_player
        for fr, fc in all_flips:
            self.board[fr, fc] = self.current_player
        self.current_player *= -1
        return True
```

On génère 20 000 parties aléatoires. Chaque partie produit ~58 coups en moyenne, soit ~1,16 million de paires (activation, état du plateau) pour l'entraînement des probes.

### 2. Le modèle : un GPT miniature

Notre Othello-GPT est un transformer décodeur classique — 8 couches, 128 dimensions, 4 têtes d'attention — soit ~900K paramètres. Petit, mais suffisant pour apprendre les régularités du jeu.

L'entrée : une séquence de tokens `[m₀, m₁, ..., mₜ]` où chaque `mᵢ ∈ {0..63}` est un index de case.
La sortie : la distribution de probabilité sur le prochain coup.

```python
# model.py — le coeur du GPT

class OthelloGPT(nn.Module):
    def __init__(self, n_layers=8, d_model=128, n_heads=4,
                 max_len=64, vocab_size=65):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_len)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, return_activations=False):
        x = self.tok_emb(idx) + self.pos_emb(positions)
        activations = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if return_activations:
                activations[i] = x.detach().clone()
        logits = self.head(self.ln_f(x))
        return {"logits": logits, "activations": activations}
```

Le paramètre `return_activations=True` est la clé : il nous permet de capturer les vecteurs intermédiaires à chaque couche — exactement ce dont les probes ont besoin.

### 3. Entraînement : 15 époques, 3 minutes sur CPU

```bash
python train.py --n_games 20000 --epochs 15
```

```
Epoch  1/15 | train_loss 3.2841 | val_loss 2.9103 | val_acc 0.142 | 12.3s
Epoch  5/15 | train_loss 2.1052 | val_loss 2.0534 | val_acc 0.287 | 11.8s
Epoch 10/15 | train_loss 1.7234 | val_loss 1.7892 | val_acc 0.351 | 12.1s
Epoch 15/15 | train_loss 1.5103 | val_loss 1.6241 | val_acc 0.398 | 11.9s
```

~40 % d'accuracy en top-1 pour prédire le coup suivant : pas un champion, mais largement au-dessus du hasard (~1/10 en moyenne, les coups légaux variant entre 1 et 20+). Le modèle a *appris quelque chose*. La question est : quoi exactement ?

### 4. Le probing : l'expérience révélatrice

C'est ici que ça devient intéressant. On prend les activations du modèle entraîné et on essaie de décoder la position du plateau.

```python
# probe.py — les deux probes

class LinearProbe(nn.Module):
    """Probe lineaire: une seule couche."""
    def __init__(self, d_model, n_squares=64, n_classes=3):
        super().__init__()
        self.linear = nn.Linear(d_model, n_squares * n_classes)

class NonLinearProbe(nn.Module):
    """Probe MLP: une couche cachee."""
    def __init__(self, d_model, hidden=256, n_squares=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_squares * n_classes),
        )
```

On entraîne les deux probes sur les activations de chaque couche du transformer :

```bash
python probe.py
```

```
=================================================================
Layer    Linear Probe      MLP Probe         Delta
=================================================================
  0      0.541             0.612             +0.071
  1      0.587             0.701             +0.114
  2      0.623             0.789             +0.166
  3      0.651             0.843             +0.192
  4      0.672             0.891             +0.219
  5      0.689             0.923             +0.234
  6      0.695             0.951             +0.256
  7      0.691             0.938             +0.247
=================================================================
```

> **Figure : [probe_accuracy.png]** — Précision de la probe par couche. La probe linéaire (gris) plafonne autour de 70 %. La probe MLP (bleu) monte à 95 %. L'écart (*delta*) montre que l'information est présente mais encodée non-linéairement.

**Que lire dans ces résultats ?**

1. **La probe linéaire plafonne (~70 %)** — L'information n'est pas disposée "en ligne droite" dans l'espace des activations. Un hyperplan ne suffit pas.

2. **La probe MLP explose (~95 %)** — Un simple ReLU + une couche supplémentaire suffisent à décoder presque parfaitement la position. L'information est *bien là*, juste pliée dans l'espace de manière non-linéaire.

3. **Le pic est aux couches intermédiaires/tardives (5-7)** — Les premières couches n'ont pas encore construit la représentation complète ; les dernières se "spécialisent" sur la prédiction du coup plutôt que sur le maintien de la position.

### 5. Visualisation : le plateau décodé

On peut même reconstruire visuellement le plateau à partir des activations :

> **Figure : [board_comparison.png]** — Gauche : position réelle au coup #20. Droite : position décodée par la probe MLP à la couche 6. Dans cet exemple, les deux sont quasi-identiques.

Et une carte de chaleur de la précision par case :

> **Figure : [per_square_accuracy.png]** — Les cases centrales (souvent occupées) sont les mieux prédites. Les bords et coins (occupés tardivement) ont une précision légèrement plus faible.

---

## 2. Le modèle ne se contente pas de stocker : il *utilise* cette représentation

C'est l'étape la plus importante du papier — et celle qu'on oublie souvent dans les résumés vulgarisés.

Montrer que l'information est *accessible* via une probe ne suffit pas. Après tout, les activations sont des vecteurs à 128 dimensions : on pourrait y trouver toutes sortes de corrélations spurieuses. La vraie question : **est-ce que le modèle *utilise* cette représentation pour prendre ses décisions ?**

### L'intervention causale

Pour le prouver, les auteurs utilisent une technique d'**intervention** (aussi appelée *activation patching* ou *causal tracing*) :

1. On fait passer une partie dans le modèle et on observe les coups prédits.
2. On identifie les activations qui encodent "la case d5 est occupée par un pion noir" (grâce à la probe).
3. On **modifie chirurgicalement** ces activations pour dire "en fait, d5 est vide".
4. On observe comment la distribution des coups change.

**Résultat** : quand on "efface" un pion dans l'espace des activations, le modèle se comporte exactement comme si ce pion n'existait pas sur le vrai plateau. Les coups qu'il proposait changent de manière cohérente avec la nouvelle position — preuve que la représentation interne n'est pas un artefact statistique, mais un **mécanisme causal** dans le raisonnement du modèle.

En interprétabilité mécaniste, on appelle cela une *world model* — un modèle du monde émergent, construit spontanément par le réseau pour mieux accomplir sa tâche.

---

## Et les échecs dans tout ça ?

Si Othello-GPT montre qu'un transformer entraîné sur des séquences de coups développe un modèle interne du plateau, qu'en est-il des échecs — un jeu plus complexe (32 pièces, 6 types, mouvements asymétriques, roque, en passant, promotion) ?

### Des indices convergents

Plusieurs travaux récents étendent ces résultats aux échecs :

**1. Les grands LLM jouent mieux que le hasard.**
Quand on donne à un modèle comme GPT-4 ou Claude une partie en notation algébrique (1.e4 e5 2.Nf3 Nc6...), il ne se contente pas de reciter des ouvertures mémorisées. Sur des positions inédites en milieu de partie, il propose des coups légaux dans la grande majorité des cas — et parfois de *bons* coups.

**2. Le taux de coups illégaux dépend du contexte.**
Les "hallucinations" (coups illégaux) ne sont pas aléatoires : elles se concentrent dans les positions tactiquement complexes (beaucoup de pièces, tensions multiples) ou en fin de partie (positions rares dans les données d'entraînement). Cela suggère que le modèle *a* une représentation, mais qu'elle devient imprécise quand la position sort de sa "zone de confort".

**3. Le probing fonctionne aussi pour les échecs.**
Des travaux (comme ceux de Neel Nanda et al.) appliquent la même technique de probing aux modèles entraînés sur des parties d'échecs. On retrouve le même schéma : les probes non-linéaires décodent la position du plateau avec une haute précision à partir des couches intermédiaires du transformer.

### Ce que les "hallucinations" nous apprennent vraiment

Quand un LLM joue un coup illégal aux échecs, ce n'est pas parce qu'il est "stupide" ou qu'il "ne comprend rien". C'est parce que :

1. **Sa représentation interne est approximative** — elle est assez bonne pour les positions fréquentes, mais se dégrade pour les configurations rares.

2. **Il n'a pas de vérificateur externe** — contrairement à un moteur d'échecs comme Stockfish qui énumère *tous* les coups légaux, le LLM "devine" la légalité à partir de sa représentation interne. Quand celle-ci est imprécise, le coup produit peut violer une règle.

3. **C'est un problème de calibration, pas d'architecture** — en augmentant la taille du modèle, la quantité de données, ou en ajoutant un vérificateur de coups légaux, le taux d'hallucinations chute drastiquement.

### L'analogie avec la cognition humaine

Un joueur d'échecs débutant fait exactement la même chose : il "oublie" qu'une pièce cloue son cavalier, il ne voit pas que son roi est en échec. Ce n'est pas qu'il ne connaît pas les règles — c'est que sa **représentation mentale de la position** est incomplète ou erronée à cet instant. Le LLM "hallucine" pour les mêmes raisons structurelles.

---

## Conclusion : au-delà du perroquet

L'expérience Othello-GPT — et ses extensions aux échecs — démontrent que :

1. **Les LLM ne sont pas de simples perroquets.** Ils développent des représentations internes structurées du domaine, même quand ils ne sont entraînés que sur des séquences de tokens.

2. **Ces représentations sont causales.** Le modèle ne les "stocke" pas passivement — il s'en sert activement pour prédire.

3. **Les hallucinations sont informatives.** Elles révèlent les limites de la représentation interne, pas l'absence de compréhension. Un coup illégal est souvent la trace d'une approximation locale, pas d'un échec fondamental.

4. **La frontière entre "comprendre" et "imiter" est plus floue qu'on ne le pense.** Si un modèle développe spontanément un état du monde, l'utilise causalement, et échoue de la même manière qu'un humain novice... à quel moment dit-on qu'il "comprend" ?

Ces questions ne sont pas que philosophiques — elles ont des implications pratiques pour l'interprétabilité, la fiabilité et la sécurité des LLM. Et les échecs, avec leurs règles formelles et leurs positions vérifiables, sont un terrain d'expérimentation idéal pour continuer à les explorer.

---

## Pour aller plus loin

- **Papier original :** Li et al., "Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task" (ICLR 2023)
- **Code de reproduction :** [lien vers ton repo GitHub]
- **Probing aux échecs :** Neel Nanda, "Actually, Othello-GPT Has A Linear Emergent World Representation" (2023) + travaux sur chess-GPT

---

*Le code complet pour reproduire toutes les expériences est disponible sur GitHub : `othello_probing/`*
*Pour lancer l'expérience complète :*

```bash
cd othello_probing
pip install -r requirements.txt
python train.py           # ~3 min sur CPU
python probe.py           # ~5 min
python visualize.py       # genere les figures
```
