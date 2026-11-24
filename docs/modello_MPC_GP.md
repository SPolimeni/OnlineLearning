# Modello MPC con correzione dell'uscita basata su Gaussian Process

## 1. Descrizione generale

Il modello **MPC-GP** mantiene la formulazione non lineare del controllore MPC di base, ma corregge online le uscite previste dal modello **NNARX** mediante un **Gaussian Process (GP)**.

L'idea centrale è separare:

1. il **modello nominale NNARX**, addestrato inizialmente e mantenuto fisso;
2. il **modello GP dell'errore di previsione**, aggiornato online con le misure più recenti dell'impianto;
3. il **problema MPC**, che utilizza come uscita finale la somma tra previsione NNARX e correzione GP.

La previsione impiegata dal controllore è quindi

\[
\tilde y_{h+1|k}=y_{h+1|k}+\hat e_{h+1|k},
\]

dove `y` è l'uscita nominale del NNARX, `e_hat` è la correzione stimata dal GP e `y_tilde` è l'uscita corretta usata nella funzione obiettivo e nei vincoli dell'MPC.

Questa architettura permette di compensare online errori sistematici, deriva del processo e disallineamenti tra modello nominale e impianto senza riaddestrare il NNARX e senza interrompere il funzionamento dell'impianto.

---

## 2. Indici e orizzonti

| Simbolo | Descrizione |
|---|---|
| `k` | Istante corrente dell'MPC e del GP. |
| `h` | Indice di previsione, con `h = k, ..., k + H - 1`. |
| `H` | Lunghezza dell'orizzonte di previsione dell'MPC. |
| `τ` | Indice dei campioni passati nel regressore NNARX. |
| `n_τ` | Ordine del regressore NNARX, ossia numero di ritardi ingresso-uscita. |
| `n_e` | Numero di errori passati inseriti nel regressore GP. |
| `n_u` | Numero di ingressi passati inseriti nel regressore GP. |
| `n_y` | Numero di uscite passate inserite nel regressore GP. |
| `k_max` | Numero massimo di campioni recenti conservati nel dataset del GP. |

Nell'esempio mostrato, per semplicità, si assume

\[
n_e=n_u=n_y=n_\tau,
\]

ma i quattro ordini possono essere scelti diversamente.

---

## 3. Errore di previsione appreso dal GP

L'errore del modello NNARX all'istante `k` è definito come

\[
e_k=y_k^p-y_k\in\mathbb{R}^{n_y},
\]

dove:

| Simbolo | Descrizione |
|---|---|
| `y_k^p` | Uscita misurata sull'impianto. |
| `y_k` | Uscita prevista dal modello NNARX nominale. |
| `e_k` | Errore di previsione, calcolato come misura meno previsione nominale. |
| `n_y` | Dimensione del vettore di uscita. |

Il GP non sostituisce il NNARX. Il GP apprende la dinamica residua che lega errori, ingressi e uscite passati all'errore di previsione a un passo futuro.

---

## 4. Regressore del Gaussian Process

Il regressore utilizzato dal GP è

\[
\xi_k=
\begin{bmatrix}
 e_k^\top & \ldots & e_{k-n_e}^\top &
 u_k^\top & \ldots & u_{k-n_u}^\top &
 y_k^\top & \ldots & y_{k-n_y}^\top
\end{bmatrix}^\top.
\]

Il regressore contiene:

- errore corrente e errori passati del NNARX;
- ingresso corrente e ingressi passati del sistema;
- uscita nominale corrente e uscite nominali passate del NNARX.

Il target associato a `ξ_k` è l'errore a un passo futuro `e_{k+1}`. Il GP apprende quindi la relazione

\[
\xi_k \longmapsto e_{k+1}.
\]

### Dimensione del regressore

Se `e`, `u` e `y` hanno rispettivamente dimensione `n_y`, `n_v` e `n_y`, la dimensione di `ξ_k` è

\[
(n_e+1)n_y+(n_u+1)n_v+(n_y+1)n_y.
\]

In questa espressione, l'uso dello stesso simbolo `n_y` sia per la dimensione dell'uscita sia per l'ordine dei ritardi può generare ambiguità. In un'implementazione software è preferibile distinguere, ad esempio, tra:

- `n_out`: numero di uscite;
- `p_e`, `p_u`, `p_y`: ordini dei ritardi di errori, ingressi e uscite.

---

## 5. Dataset online del GP

Il dataset impiegato all'istante `k` è

\[
\mathcal D_k=
\begin{bmatrix}
\boldsymbol{\xi}_k & \boldsymbol e_k
\end{bmatrix}^\top,
\]

con

\[
\boldsymbol{\xi}_k=
\{\xi_\tau\}_{\tau=k-k_{\max},\ldots,k-1},
\]

\[
\boldsymbol e_k=
\{e_\tau^p\}_{\tau=k-k_{\max}+1,\ldots,k}.
\]

Il dataset associa quindi ciascun regressore passato `ξ_τ` all'errore osservato al passo successivo. Il limite `k_max` realizza una finestra mobile contenente al massimo i campioni più recenti.

### Ruolo di `k_max`

- un valore piccolo riduce il costo di addestramento e privilegia l'adattamento rapido alle condizioni recenti;
- un valore grande fornisce più dati, ma aumenta il costo computazionale e può includere condizioni operative non più rappresentative;
- la finestra mobile implica che i campioni più vecchi vengono eliminati quando è raggiunto il numero massimo consentito.

---

## 6. Previsione della correzione GP

La previsione a un passo dell'errore NNARX è

\[
\hat e_{k+1}=f^{GP}(\xi_k\mid\mathcal D_k),
\]

dove `f^{GP}` è la funzione media a posteriori del Gaussian Process condizionata sul dataset `D_k`.

Gli iperparametri del GP sono stimati massimizzando la **log marginal likelihood**. Gli iperparametri includono tipicamente i parametri del kernel, la varianza del segnale e la varianza del rumore osservativo, anche se il kernel specifico non è indicato nella formulazione fornita.

---

## 7. Modello di uscita corretto

Il modello NNARX nominale produce

\[
y_{k+1}=\theta_0^\top f^{NN}(\phi_k,u_k;\bar\Theta).
\]

L'uscita corretta è

\[
\tilde y_{k+1}
=
\theta_0^\top f^{NN}(\phi_k,u_k;\bar\Theta)
+f^{GP}(\xi_k\mid\mathcal D_k).
\]

In forma compatta:

\[
\tilde y_{k+1}=y_{k+1}+\hat e_{k+1}.
\]

| Variabile | Ruolo |
|---|---|
| `y_{k+1}` | Previsione nominale del NNARX. |
| `e_hat_{k+1}` | Errore previsto dal GP. |
| `y_tilde_{k+1}` | Previsione finale corretta utilizzata dall'MPC. |

---

## 8. Problema di ottimizzazione MPC-GP

La formulazione generale è

\[
\min_{v_{k:k+H-1|k}}
\sum_{h=k}^{k+H-1}
\mathcal L(v_{h|k},\tilde y_{h|k}),
\]

soggetta ai vincoli descritti nelle sezioni successive.

### Variabile decisionale principale

| Variabile | Descrizione |
|---|---|
| `v_{h|k}` | Vettore delle variabili manipolate o dei comandi all'istante previsto `h`, ottimizzati all'istante `k`. |
| `v_{k:k+H-1|k}` | Intera sequenza di controllo ottimizzata sull'orizzonte `H`. |

La composizione fisica di `v` dipende dall'applicazione. Nel modello termico di base può comprendere, ad esempio, i set-point delle temperature di mandata della caldaia elettrica e della caldaia a gas.

### Funzione di costo

| Elemento | Descrizione |
|---|---|
| `L(v_{h|k}, y_tilde_{h|k})` | Costo di stadio, funzione dei comandi e delle uscite corrette. |

La formulazione non espande `L`. Il costo può quindi includere costi energetici, penalizzazioni sul tracking, regolarizzazione dei comandi e penalizzazioni delle variabili di slack. Rispetto all'MPC nominale, la differenza essenziale è che il costo viene valutato usando `y_tilde` anziché la sola uscita NNARX `y`.

---

## 9. Variabili del modello

### 9.1 Variabili decisionali

| Variabile | Tipo | Descrizione |
|---|---|---|
| `v_{h|k}` | Decisionale | Comando ottimizzato dall'MPC. |
| Eventuali slack | Decisionali ausiliarie | Possono essere introdotte per rendere soft alcuni vincoli, ma non sono riportate esplicitamente nella formulazione. |

### 9.2 Variabili previste e dipendenti

| Variabile | Tipo | Descrizione |
|---|---|---|
| `y_{h+1|k}` | Dipendente | Uscita nominale prevista dal NNARX. |
| `e_hat_{h+1|k}` | Dipendente | Correzione dell'errore prevista dal GP. |
| `y_tilde_{h+1|k}` | Dipendente | Uscita corretta finale usata dall'MPC. |
| `φ_{h|k}` | Regressore NNARX | Contiene ingressi e uscite ritardati richiesti dal modello nominale. |
| `ξ_{h|k}` | Regressore GP | Contiene errori, ingressi e uscite ritardati. |

### 9.3 Misure e dati storici

| Variabile | Descrizione |
|---|---|
| `y_k^p` | Uscita misurata dall'impianto. |
| `u_{k-τ}` | Ingresso storico effettivamente applicato. |
| `y^p_{k+1-τ}` | Uscita storica misurata. |
| `e_{k-τ}` | Errore storico del NNARX. |
| `D_k` | Dataset online usato per addestrare o aggiornare il GP. |

### 9.4 Parametri

| Parametro | Descrizione |
|---|---|
| `Θ_bar` | Parametri addestrati e fissi della rete NNARX. |
| `θ_0` | Parametri del livello di uscita del NNARX. |
| Iperparametri GP | Parametri del kernel e del rumore, stimati tramite log marginal likelihood. |
| `H` | Orizzonte di previsione dell'MPC. |
| `n_τ` | Ordine del regressore NNARX. |
| `n_e`, `n_u`, `n_y` | Ordini del regressore GP. |
| `k_max` | Numero massimo di coppie ingresso-target nel dataset GP. |
| `y_lower`, `y_upper` | Limiti inferiori e superiori delle uscite corrette. |

---

## 10. Vincoli dell'MPC-GP

Tutti i vincoli valgono per

\[
h\in[k,k+H-1].
\]

### 10.1 Modello nominale NNARX

\[
y_{h+1|k}
=
\theta_0^\top f^{NN}(\phi_{h|k},u_{h|k};\bar\Theta).
\]

Il vincolo genera la previsione nominale della dinamica del processo. I parametri `Θ_bar` restano fissi e non vengono ri-identificati durante l'esercizio.

### 10.2 Inizializzazione della memoria NNARX

\[
u_{k-\tau|k}=u_{k-\tau},
\qquad
 y_{k+1-\tau|k}=y^p_{k+1-\tau},
\qquad
\tau=1,\ldots,n_\tau.
\]

Il regressore NNARX viene inizializzato con ingressi storici applicati e uscite storiche misurate. Questo vincolo collega la previsione futura alle condizioni reali correnti dell'impianto.

### 10.3 Correzione dell'uscita

\[
\tilde y_{h+1|k}=y_{h+1|k}+\hat e_{h+1|k}.
\]

L'uscita usata dal controllore è la somma tra la previsione nominale del NNARX e la correzione GP.

### 10.4 Predizione dell'errore mediante GP

\[
\hat e_{h+1|k}=f^{GP}(\xi_{h|k}\mid\mathcal D_k).
\]

La correzione è calcolata dalla media a posteriori del GP sulla base del regressore previsto e del dataset disponibile all'inizio dell'iterazione MPC.

### 10.5 Inizializzazione della memoria del GP

\[
e_{k-\tau|k}=e_{k-\tau},
\qquad
\tau=0,\ldots,n_\tau.
\]

Gli errori passati necessari al regressore GP sono inizializzati con gli errori effettivamente osservati prima dell'istante corrente.

### 10.6 Vincoli operativi generali

\[
\psi(v_{h|k},v_{h|k-1},\tilde y_{h|k})\le 0.
\]

La funzione vettoriale `ψ` raccoglie i vincoli operativi del problema. Può includere:

- limiti inferiori e superiori sui comandi;
- limiti sulla variazione dei comandi tra campioni consecutivi;
- relazioni fisiche e algebriche tra variabili;
- limiti di capacità degli attuatori;
- bilanci energetici o di massa;
- eventuali vincoli di comfort, sicurezza o qualità del servizio.

Poiché `ψ` dipende da `y_tilde`, i vincoli vengono verificati rispetto alla previsione corretta dal GP e non rispetto alla sola uscita nominale.

> La notazione riportata nell'immagine appare come `ψ(v_{h|k}, v_{h|k}, y_tilde_{h|k})`. È possibile che uno dei due argomenti rappresenti il comando del passo precedente o un'altra componente del vettore di controllo. La definizione esplicita di `ψ` è necessaria per risolvere questa ambiguità nell'implementazione.

### 10.7 Limiti sulle uscite corrette

\[
\underline y\le\tilde y_{h+1|k}\le\overline y.
\]

I limiti operativi vengono imposti direttamente all'uscita corretta. Questa scelta è fondamentale: l'MPC considera il disallineamento stimato dal GP quando verifica il rispetto dei limiti del processo.

---

## 11. Classificazione dei vincoli

| Categoria | Vincoli |
|---|---|
| Vincoli dinamici | Modello NNARX e propagazione del regressore. |
| Vincoli di correzione | Somma tra uscita NNARX e previsione dell'errore GP. |
| Vincoli di inizializzazione | Assegnazione della storia di ingressi, uscite misurate ed errori. |
| Vincoli operativi | Funzione vettoriale `ψ ≤ 0`. |
| Vincoli sulle uscite | `y_lower ≤ y_tilde ≤ y_upper`. |
| Vincoli sui dati del GP | Dataset limitato agli ultimi `k_max` campioni. Non è un vincolo dell'NLP, ma una regola dell'aggiornamento online. |

Nella formulazione mostrata non sono indicate variabili di slack. I limiti sulle uscite devono quindi essere considerati **hard**, salvo che l'implementazione introduca esplicitamente slack e relative penalizzazioni.

---

## 12. Sequenza operativa online

A ciascun istante `k`, l'architettura MPC-GP esegue i seguenti passaggi:

1. acquisisce la nuova uscita misurata `y_k^p`;
2. calcola l'errore del modello nominale `e_k = y_k^p - y_k`;
3. costruisce la nuova coppia regressore-target per il GP;
4. aggiorna la finestra dati `D_k`, mantenendo al massimo `k_max` campioni;
5. stima o aggiorna gli iperparametri del GP tramite massimizzazione della log marginal likelihood;
6. costruisce il problema MPC sull'orizzonte `H`;
7. calcola, per ogni passo previsto, l'uscita NNARX `y`, la correzione GP `e_hat` e l'uscita finale `y_tilde`;
8. ottimizza la sequenza `v_{k:k+H-1|k}`;
9. applica al processo soltanto il primo comando ottimo;
10. ripete la procedura al campione successivo secondo il principio di orizzonte mobile.

---

## 13. Differenze rispetto all'MPC nominale

| Aspetto | MPC nominale | MPC-GP |
|---|---|---|
| Modello principale | NNARX fisso | NNARX fisso |
| Adattamento online | Assente | GP aggiornato con dati recenti |
| Uscita usata dall'MPC | `y` | `y_tilde = y + e_hat` |
| Errore del modello | Non compensato esplicitamente | Previsto dal GP |
| Vincoli sulle uscite | Applicati a `y` | Applicati a `y_tilde` |
| Costo computazionale | Inferiore | Superiore per training e inferenza GP |
| Gestione dell'incertezza | Non esplicita | Disponibile tramite varianza GP, ma non inclusa nei vincoli mostrati |

---

## 14. Aspetti critici e informazioni mancanti

Per un'implementazione completa devono essere specificati:

1. **composizione di `v`, `u`, `y`, `φ` e `ξ`**;
2. **forma esplicita del costo `L`**;
3. **forma esplicita dei vincoli `ψ`**;
4. **kernel GP**, ad esempio squared exponential, Matérn o automatic relevance determination;
5. **normalizzazione** di regressori e target del GP;
6. **valore di `k_max`** e politica di aggiornamento della finestra;
7. **frequenza di riottimizzazione degli iperparametri**, che può essere diversa dalla frequenza dell'MPC per contenere il costo computazionale;
8. **propagazione multi-step del GP**, poiché lungo l'orizzonte alcuni elementi di `ξ_{h|k}` sono previsti e non misurati;
9. **uso della varianza predittiva**, non presente nei vincoli mostrati;
10. **gestione della fattibilità**, ad esempio tramite slack sui limiti di uscita;
11. **criteri di qualità e informatività dei dati**, perché una finestra composta da dati recenti ma poco variabili può produrre un GP scarsamente informativo;
12. **guardie numeriche**, come jitter sulla matrice di covarianza e limiti sugli iperparametri.

---

## 15. Osservazione sulla sicurezza dei vincoli

La formulazione adopera soltanto la media della correzione GP:

\[
\hat e=f^{GP}(\xi\mid\mathcal D).
\]

La varianza predittiva del GP non compare nei limiti su `y_tilde`. Di conseguenza, il controllore corregge il bias atteso, ma non crea automaticamente un margine di sicurezza nelle regioni in cui il GP è incerto.

Una possibile estensione robusta o *chance-constrained* consiste nel restringere i limiti mediante la deviazione standard predittiva `σ_GP`, ad esempio:

\[
\underline y+\beta\sigma_{GP}
\le
\tilde y
\le
\overline y-\beta\sigma_{GP},
\]

dove `β` determina il livello di prudenza. Questa relazione è un'estensione progettuale e non fa parte della formulazione originale mostrata.

---

## 16. Sintesi

Il modello MPC-GP è un controllore predittivo non lineare con due livelli di modellazione:

- il **NNARX** rappresenta la dinamica nominale del processo;
- il **Gaussian Process** apprende online l'errore residuo del NNARX.

Le variabili decisionali sono i comandi `v` sull'orizzonte. Le grandezze `y`, `e_hat` e `y_tilde` sono variabili dipendenti generate rispettivamente dal NNARX, dal GP e dalla loro somma. I vincoli comprendono l'evoluzione del modello nominale, l'inizializzazione delle memorie, la correzione GP, i vincoli operativi generali `ψ` e i limiti sulle uscite corrette.

Il vantaggio principale è l'adattamento online senza riaddestramento del modello nominale. I principali punti critici sono il costo computazionale del GP, la selezione di dati informativi, la propagazione multi-step della correzione e la necessità di considerare esplicitamente l'incertezza predittiva per evitare violazioni dei vincoli reali.
