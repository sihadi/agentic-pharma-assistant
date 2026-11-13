# Rapport d'exploration (Phase 0)

Exploration réalisée **avant** l'implémentation métier, puis complétée par les
mesures faites pendant l'ingestion. Chaque constat ci-dessous a orienté une
décision de conception (colonne « → »).

## 1. Ressources inspectées

| Ressource | Contenu | Usage |
|---|---|---|
| `assignment-pharma/assignment-pharma.pdf` | Assignment officiel Syntax (6 p.) | Exigences contractuelles (Plan, Retrieve, Tools ≥ 2, Critic, Answer, `/health`, `/ask`, Docker, frontend) |
| `Cahier_des_Charges_Agent_Pharma_Syntax.pdf` | Cahier des charges v1 (7 p.) | Ordre de réalisation, format JSON du critic, disclaimer imposé, 4 questions |
| `Cahier_des_charges_..._Assistant.pdf` | Cahier des charges technique (15 p.) | Workflow cible en 9 étapes, contrat des outils, traçabilité, plan de tests T-01…T-20 |
| `assignment-pharma/corpus/` (→ `data/raw/`) | 21 fichiers Markdown | Corpus |
| `assignment-pharma/.env` | Variables d'accès au proxy LiteLLM | Noms de variables uniquement (la clé n'est jamais recopiée) |

## 2. Inventaire du corpus

| Catégorie | Fichiers | Remarque |
|---|---:|---|
| Labels DailyMed (FDA) | 10 | 2 à 76 Ko ; source clinique de référence |
| Monographies MedlinePlus | 10 | 7 à 15 Ko ; langage patient |
| `GLOSSARY.md` | 1 | Définitions générales |
| **Total** | **21** | 460 Ko, UTF-8, aucun fichier vide ni doublon |

10 médicaments, chacun avec un label DailyMed et une monographie MedlinePlus :
acetaminophen, albuterol, amoxicillin, apixaban, aspirin, atorvastatin,
ibuprofen, metformin, sertraline, warfarin.

## 3. Structure des documents — trois conventions de titres

1. **Labels FDA au format PLR** (warfarin, apixaban, metformin, sertraline,
   amoxicillin, atorvastatin) : titres numérotés `## 4 CONTRAINDICATIONS`,
   sous-sections `### 5.3 …` jusqu'au niveau `#####`.
2. **Labels anciens / OTC** (ibuprofen, albuterol, acetaminophen, aspirin) :
   titres non numérotés (`CONTRAINDICATIONS`, `Uses`, `Directions`,
   `Warnings`). Les OTC n'ont pas de section *Contraindications* : l'équivalent
   est un paragraphe « Do not use » à l'intérieur de *Warnings*.
3. **MedlinePlus** : titres en questions (`What special precautions should I
   follow?`), identiques pour les 10 fiches, plus `Brand names`.

→ **Taxonomie canonique** (`app/ingestion/taxonomy.py`) : 21 clés stables
(`contraindications`, `drug_interactions`, …) ; les numéros PLR suivent
l'ordre réglementaire 21 CFR 201.57, les autres titres sont mappés par règles.

## 4. Bruit mesuré dans les données brutes

| Bruit | Occurrences | Exemple | → Règle de nettoyage |
|---|---:|---|---|
| Exposant perdu (m²) | 23 | `1.73 m 2is` | `unit_m2`, `unit_m2_superscript` |
| Indice perdu (B12) | 11 | `B 12Deficiency` | `vitamin_b12(_glued)` |
| Balises HTML | 170 | `Coumadin<sup>®</sup>` | `html_sup_sub` |
| Échappements Markdown | 8 | `\[INR\]` | `markdown_unescape` |
| Libellé collé | 31 | `Liver warning:This` | `colon_spacing` |
| Puces Unicode en début de ligne | 5 | `• take 2 tablets` | `bullet_normalization` |
| BOM invisibles en plein texte | 3 | `WARNINGS ﻿ and` | `zero_width_chars` |

→ Nettoyage **conservateur** : on répare la typographie, jamais les mots, les
nombres ou les négations. Chaque substitution est comptée par document dans
`data/manifest.json`. Les mots collés comme `haveliver` ne sont **pas**
corrigés : il faudrait un dictionnaire, avec un risque d'altérer le texte
médical.

## 5. Particularités découvertes

- **Boxed warning sans titre** : le label ibuprofen place son avertissement
  encadré entre le titre et la première section. → Pour DailyMed, ce
  préambule est classé `boxed_warning`.
- **Le label « albuterol » est un produit combiné** (ipratropium bromide +
  albuterol sulfate). L'agent doit donc citer les termes exacts du label.
- **Noms de marque** dans les fiches MedlinePlus (`Eliquis`, `Coumadin`,
  `Zoloft`…). → Table d'alias **extraite du corpus** (rien n'est codé en dur),
  ce qui permet de résoudre « Coumadin » en warfarin.
- **Tableaux** à pipes (dosage selon génotype, interactions) : jusqu'à 76
  lignes par label. → Un tableau n'est jamais coupé de son en-tête.
- **Paragraphes de plus de 4 000 caractères** (atorvastatin, sertraline). →
  Découpage de repli par lignes, puis par phrases.

## 6. Configuration d'accès : deux pièges détectés

- Le `.env` fourni nomme la variable **`OPENAI_BASE_UR`** (sans `L`). Syntax
  injectera « son propre .env » pour l'évaluation : le code accepte les deux
  orthographes, sinon l'application ne démarre pas.
- Les valeurs sont entourées de **guillemets simples**. `python-dotenv` les
  retire, mais `docker run --env-file` les transmet tels quels. → Le chargement
  de la configuration les retire, sinon le nom du modèle serait invalide dans
  le conteneur.
- Capacités du proxy vérifiées : embeddings (1536 dimensions), chat,
  `temperature=0`, **tool calling natif parallèle**, **structured outputs
  `json_schema` strict**. Le proxy renvoie des `429` (cooldown) en rafale.
  → Retries avec backoff exponentiel.

## 7. Défis pour le chunking et le retrieval

1. Conserver l'attribution médicament / section : un chunk « Pregnancy » doit
   rester rattaché à *warfarin › contre-indications*.
2. Ne jamais couper une négation, une condition ou une unité ; garder une liste
   attachée à sa phrase d'introduction (« contraindicated in patients with: »).
3. Termes exacts critiques (`eGFR`, `INR`, `MAOI`, `pimozide`, seuils) mal
   servis par les seuls embeddings.
4. Questions multi-hop : une interaction peut être documentée dans le label de
   l'*autre* médicament (le label warfarin cite sertraline nommément).

## 8. Opportunités d'innovation retenues

Chunking structurel avec en-tête contextuel, retrieval hybride dense + BM25
fusionné par RRF, outils complémentaires choisis par function calling natif,
critic à deux couches (audit déterministe avec droit de veto, plus revue LLM
affirmation par affirmation), citations au niveau de l'affirmation, lignage
complet des données et trace structurée diffusée en direct dans l'interface.
