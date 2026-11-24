# Modello MPC base per la gestione di una stazione di produzione termica

## 1. Scopo del controllore

Il modello è un **Model Predictive Control (MPC)** che determina, a ogni istante di campionamento `k`, i profili ottimali delle temperature di mandata di:

- **caldaia elettrica** (`EB`, *Electric Boiler*);
- **caldaia a gas** (`GB`, *Gas Boiler*).

L'ottimizzazione è svolta su un orizzonte di previsione di lunghezza `H`. Il modello della rete termica inserito nel controllore è una rete neurale **NNARX** già addestrata, i cui parametri restano fissi durante l'ottimizzazione.

Il controllore minimizza il costo di produzione del calore e, contemporaneamente, penalizza lo scostamento della temperatura di mandata al carico al termine dell'orizzonte rispetto al riferimento nominale `T*`.

---

## 2. Indici e orizzonti

| Simbolo | Descrizione |
|---|---|
| `k` | Istante corrente di MPC. |
| `h` | Indice di previsione, con `h = k, ..., k + H - 1`. |
| `H` | Lunghezza dell'orizzonte di previsione. |
| `H_b` | Lunghezza del blocco di controllo (*move blocking*), con `H_b << H`. |
| `τ` | Indice dei campioni passati usati dal regressore NNARX. |
| `n_τ` | Numero di ritardi ingresso-uscita del modello NNARX. |

---

## 3. Variabili decisionali

Le variabili decisionali esplicite sono le sequenze delle temperature di mandata delle due caldaie:

\[
\mathbf{T}^{s,EB}_{k:k+H-1|k}
\qquad\text{e}\qquad
\mathbf{T}^{s,GB}_{k:k+H-1|k}.
\]

In particolare:

| Variabile | Unità tipica | Descrizione |
|---|---:|---|
| `T^{s,EB}_{h|k}` | °C o K | Set-point previsto della temperatura di mandata della caldaia elettrica all'istante `h`, calcolato a `k`. |
| `T^{s,GB}_{h|k}` | °C o K | Set-point previsto della temperatura di mandata della caldaia a gas all'istante `h`, calcolato a `k`. |

Le variabili sono soggette a limiti operativi, limiti sulla variazione tra campioni consecutivi e vincoli di *move blocking*.

---

## 4. Variabili dipendenti e grandezze previste

Le seguenti quantità non sono comandi indipendenti, ma sono calcolate a partire dalle variabili decisionali e dal modello della rete:

| Variabile | Descrizione |
|---|---|
| `y_{h+1|k}` | Vettore di uscita previsto dal modello NNARX. Include le grandezze della rete necessarie all'MPC. |
| `u_{h|k}` | Vettore di ingresso del modello NNARX, costruito a partire dai comandi e dalle altre grandezze esogene definite nel modello. |
| `φ_{h|k}` | Regressore NNARX, formato da ingressi e uscite ritardati. |
| `T^{s,0}_{h|k}` | Temperatura di mandata risultante nel ramo comune/miscelato della stazione termica. |
| `T^{r,0}_{h|k}` | Temperatura di ritorno della rete alla stazione termica. |
| `T^{s,4}_{h|k}` | Temperatura di mandata presso il carico 4. |
| `q^0_{h|k}` | Portata totale d'acqua nella stazione. |
| `q^{EB}_{h|k}` | Portata d'acqua attraverso la caldaia elettrica. |
| `q^{GB}_{h|k}` | Portata d'acqua attraverso la caldaia a gas. |
| `P^{EB}_{h|k}` | Potenza termica prodotta dalla caldaia elettrica. |
| `P^{GB}_{h|k}` | Potenza termica prodotta dalla caldaia a gas. |

---

## 5. Parametri

| Parametro | Descrizione |
|---|---|
| `c_h^{EB}` | Prezzo dell'energia elettrica previsto all'istante `h`. |
| `c^{GB}` | Prezzo del gas. |
| `c_T` | Peso della penalizzazione sullo scostamento della temperatura al carico. |
| `η^{EB}` | Efficienza di conversione della caldaia elettrica. |
| `η^{GB}` | Efficienza di conversione della caldaia a gas. |
| `T_s` | Durata del passo di campionamento. |
| `T*` | Valore nominale desiderato della temperatura di mandata al carico 4. |
| `c_w` | Capacità termica specifica dell'acqua, coerente con le unità di portata e potenza adottate. |
| `q^{EB}` | Portata assegnata alla caldaia elettrica, usata nel bilancio di miscelazione. |
| `q^0_{h|k}` | Portata totale prevista; determina la portata nella caldaia a gas tramite il bilancio di massa. |
| `Θ̄` | Parametri addestrati e fissati della rete NNARX. |
| `y^p_{k+1-τ}` | Uscita passata misurata sull'impianto. |
| `ΔT̄` | Massima variazione ammessa del set-point tra due passi consecutivi. |
| `P̲^{EB}, P̄^{EB}` | Limite inferiore e superiore della potenza termica della caldaia elettrica. |
| `P̲^{GB}, P̄^{GB}` | Limite inferiore e superiore della potenza termica della caldaia a gas. |
| `T̲, T̄` e `q̲, q̄` | Limiti operativi inferiori e superiori delle rispettive temperature e portate. |

> **Nota:** la formulazione mostrata usa `q^{EB}` senza indice temporale nel bilancio di portata e nella miscelazione. Questo indica che la portata della caldaia elettrica è trattata come assegnata o costante, non come variabile decisionale dell'MPC base.

---

## 6. Funzione obiettivo

Il problema di ottimizzazione è:

\[
\min_{\mathbf{T}^{s,EB}_{k:k+H-1|k},\,\mathbf{T}^{s,GB}_{k:k+H-1|k}}
\sum_{h=k}^{k+H-1}
\left(
 c_h^{EB}\frac{P^{EB}_{h|k}T_s}{\eta^{EB}}
 +c^{GB}\frac{P^{GB}_{h|k}T_s}{\eta^{GB}}
\right)
+c_T\left(T^{s,4}_{k+H-1|k}-T^*\right)^2.
\]

La funzione obiettivo comprende:

1. **costo elettrico della caldaia elettrica**, calcolato dall'energia termica prodotta e dall'efficienza di conversione;
2. **costo del gas della caldaia a gas**, anch'esso corretto mediante l'efficienza;
3. **penalizzazione terminale**, applicata allo scostamento tra la temperatura di mandata al carico 4 nell'ultimo istante previsto e il riferimento `T*`.

La penalizzazione è terminale, non distribuita su tutto l'orizzonte nella formulazione mostrata.

---

## 7. Vincoli

Tutti i vincoli seguenti valgono per `h ∈ [k, k + H - 1]`, salvo diversa indicazione.

### 7.1 Modello dinamico NNARX

\[
y_{h+1|k}=\theta_0^{\top}f^{NN}(\phi_{h|k};\bar{\Theta}).
\]

Il vincolo propaga le uscite previste della rete termica. `f^{NN}` rappresenta la parte non lineare della rete, `θ_0` il livello di uscita e `Θ̄` i parametri addestrati mantenuti fissi.

### 7.2 Inizializzazione del regressore

\[
u_{k-\tau|k}=u_{k-\tau},
\qquad
y_{k+1-\tau|k}=y^p_{k+1-\tau},
\qquad \tau=1,\ldots,n_\tau.
\]

I campioni passati del regressore sono inizializzati con gli ingressi storici applicati e con le uscite misurate dall'impianto. La struttura NNARX rende quindi utilizzabili direttamente le misure passate, senza introdurre esplicitamente un osservatore di stato.

### 7.3 Limiti sui comandi

\[
\underline{T}^{s,EB}\le T^{s,EB}_{h|k}\le\overline{T}^{s,EB},
\]

\[
\underline{T}^{s,GB}\le T^{s,GB}_{h|k}\le\overline{T}^{s,GB}.
\]

Questi vincoli impongono gli intervalli operativi ammessi alle temperature di mandata delle due caldaie.

### 7.4 Miscelazione delle portate

\[
T^{s,0}_{h|k}=
\frac{q^{EB}T^{s,EB}_{h|k}+q^{GB}_{h|k}T^{s,GB}_{h|k}}
{q^0_{h|k}}.
\]

La temperatura di mandata comune è ottenuta mediante una media pesata sulle portate dei due generatori.

### 7.5 Bilancio di massa

\[
q^{GB}_{h|k}=q^0_{h|k}-q^{EB}.
\]

La portata nella caldaia a gas è la differenza tra la portata totale e quella assegnata alla caldaia elettrica.

### 7.6 Potenza termica delle caldaie

\[
P^{EB}_{h|k}=c_w q^{EB}
\left(T^{s,EB}_{h|k}-T^{r,0}_{h|k}\right),
\]

\[
P^{GB}_{h|k}=c_w q^{GB}_{h|k}
\left(T^{s,GB}_{h|k}-T^{r,0}_{h|k}\right).
\]

Le potenze termiche sono calcolate dal prodotto tra capacità termica dell'acqua, portata e salto termico tra mandata e ritorno.

### 7.7 Limiti di potenza

\[
\underline{P}^{EB}\le P^{EB}_{h|k}\le\overline{P}^{EB},
\]

\[
\underline{P}^{GB}\le P^{GB}_{h|k}\le\overline{P}^{GB}.
\]

I vincoli rappresentano la modulazione minima e la capacità massima dei generatori.

### 7.8 Limiti sulla velocità di variazione dei comandi

\[
-\overline{\Delta T}\le
T^{s,EB}_{h+1|k}-T^{s,EB}_{h|k}
\le\overline{\Delta T},
\]

\[
-\overline{\Delta T}\le
T^{s,GB}_{h+1|k}-T^{s,GB}_{h|k}
\le\overline{\Delta T}.
\]

Questi vincoli limitano le variazioni dei set-point tra passi consecutivi, riducendo comandi troppo rapidi o oscillanti.

### 7.9 Move blocking

\[
T^{s,EB}_{h|k}=T^{s,EB}_{h^*|k},
\qquad
h^*=\left\lfloor\frac{h-k}{H_b}\right\rfloor H_b+k,
\]

\[
T^{s,GB}_{h|k}=T^{s,GB}_{h^*|k},
\qquad
h^*=\left\lfloor\frac{h-k}{H_b}\right\rfloor H_b+k.
\]

I comandi rimangono costanti all'interno di blocchi di `H_b` passi. La tecnica riduce il numero effettivo di gradi di libertà da circa `2H` a circa `2⌈H/H_b⌉`, diminuendo il costo computazionale dell'ottimizzazione.

### 7.10 Limiti sulle uscite previste

\[
\underline{T}^{s,4}\le T^{s,4}_{h+1|k}\le\overline{T}^{s,4},
\]

\[
\underline{T}^{r,0}\le T^{r,0}_{h+1|k}\le\overline{T}^{r,0},
\]

\[
\underline{q}^{0}\le q^{0}_{h+1|k}\le\overline{q}^{0}.
\]

La formulazione originale specifica che questi limiti sono implementati con **variabili di slack**, non riportate esplicitamente nelle equazioni mostrate. Di conseguenza, i limiti sulle uscite sono verosimilmente vincoli soft per preservare la fattibilità del problema. Per una definizione completa del problema numerico occorre specificare:

- le slack associate a ciascun limite;
- il segno e i limiti delle slack, normalmente non negative;
- la relativa penalizzazione nella funzione obiettivo.

---

## 8. Classificazione sintetica

| Categoria | Elementi |
|---|---|
| Variabili manipolate | `T^{s,EB}_{h|k}`, `T^{s,GB}_{h|k}`. |
| Uscite controllate | Principalmente `T^{s,4}_{h|k}`; inoltre `T^{r,0}_{h|k}` e `q^0_{h|k}` sono mantenute entro limiti operativi. |
| Variabili dipendenti algebriche | `T^{s,0}_{h|k}`, `q^{GB}_{h|k}`, `P^{EB}_{h|k}`, `P^{GB}_{h|k}`. |
| Stato/regressore del predittore | `φ_{h|k}`, costruito con `n_τ` campioni passati di `u` e `y`. |
| Disturbi o ingressi esogeni | Le componenti di `u` non manipolate dall'MPC e le condizioni operative previste; la loro composizione non è specificata nell'estratto. |
| Vincoli hard | Limiti sui comandi, potenze, ratei e relazioni fisiche/algebriche. |
| Vincoli presumibilmente soft | Limiti su `T^{s,4}`, `T^{r,0}` e `q^0`, per i quali il testo menziona slack omesse. |

---

## 9. Struttura operativa dell'MPC

A ogni campione `k`, il controllore:

1. acquisisce gli ultimi `n_τ` ingressi e le ultime `n_τ` uscite misurate;
2. inizializza il regressore NNARX;
3. prevede l'evoluzione della rete sull'orizzonte `H`;
4. ottimizza i set-point delle temperature di mandata delle due caldaie;
5. rispetta limiti termici, limiti di potenza, ratei e *move blocking*;
6. applica al processo soltanto il primo comando ottimo;
7. al campione successivo aggiorna le misure e ripete l'ottimizzazione secondo il principio di orizzonte mobile.

---

## 10. Aspetti da chiarire per l'implementazione

L'estratto consente di ricostruire la struttura del modello, ma non contiene tutti i dati necessari per un'implementazione univoca. Devono essere definiti separatamente:

- composizione e ordine dei vettori `u`, `y` e `φ`;
- valore di `H`, `H_b`, `T_s` e `n_τ`;
- valori numerici di tutti i limiti;
- unità di misura e conversioni impiegate nella funzione di costo;
- previsione del prezzo elettrico `c_h^{EB}` e valore del prezzo del gas `c^{GB}`;
- origine o modello della portata totale `q^0_{h|k}`;
- valore e natura della portata `q^{EB}`;
- definizione completa delle variabili di slack e dei rispettivi pesi;
- gestione del vincolo di rateo al bordo dell'orizzonte, cioè il collegamento tra il primo comando previsto e l'ultimo comando effettivamente applicato;
- eventuale imposizione di `P ≥ 0` qualora i limiti inferiori riportati non la garantiscano già.

Questi elementi sono necessari per tradurre la formulazione matematica in un problema implementabile, ad esempio in Pyomo, CasADi oppure in un solutore NLP compatibile con il modello NNARX.
