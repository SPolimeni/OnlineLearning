# Modello MPC con correzione bayesiana e apprendimento attivo sicuro

## 1. Descrizione generale

Il modello combina un **Model Predictive Control non lineare**, un modello dinamico nominale **NNARX** e un meccanismo di **Bayesian Last-Layer Learning (BLL)**. L'obiettivo è adattare online il modello alle condizioni reali dell'impianto, mantenendo il rispetto dei vincoli operativi.

L'architettura, indicata come **MPC-BLL**, presenta tre caratteristiche principali:

1. il modello NNARX fornisce la previsione nominale della dinamica;
2. l'ultimo strato del modello viene aggiornato online con una regressione lineare bayesiana;
3. l'MPC alterna una modalità di **esplorazione sicura**, finalizzata a ridurre l'incertezza del modello, e una modalità di **goal reaching**, finalizzata a minimizzare il costo operativo.

A differenza della correzione mediante Gaussian Process, non viene addestrato un modello separato dell'errore. La correzione bayesiana agisce direttamente sui parametri dell'ultimo strato del modello NNARX.

---

## 2. Struttura del modello NNARX

Il modello nominale può essere scritto come

\[
y_{k+1}=\theta_0^\top f^{NN}(\phi_k,u_k;\bar\Theta),
\]

dove:

| Simbolo | Descrizione |
|---|---|
| `y_{k+1}` | Vettore di uscita previsto al passo successivo. |
| `u_k` | Vettore di ingresso del modello. |
| `φ_k` | Regressore NNARX costruito con ingressi e uscite ritardati. |
| `f^{NN}` | Mappa non lineare implementata dagli strati interni della rete. |
| `Θ̄` | Parametri degli strati interni, mantenuti fissi durante l'adattamento online. |
| `θ_0` | Parametri dell'ultimo strato lineare del modello nominale. |

Gli strati interni del NNARX estraggono quindi un vettore di caratteristiche non lineari, mentre l'ultimo strato combina linearmente tali caratteristiche per generare le uscite.

---

## 3. Parametri bayesiani dell'ultimo strato

Per ciascuna uscita `j = 1, ..., n_y`, si considera separatamente il vettore dei parametri dell'ultimo strato

\[
\theta_{0,j}\in\mathbb{R}^{n_f},
\]

dove `n_f` è il numero di caratteristiche prodotte dall'ultimo strato nascosto, includendo eventualmente il termine di bias.

I parametri incerti sono descritti da una distribuzione gaussiana:

\[
\theta_{0,j}\sim\mathcal N(\bar\theta_{0,j},\Lambda_{0,j}^{-1}),
\]

dove:

| Simbolo | Descrizione |
|---|---|
| `θ_{0,j}` | Vettore casuale dei parametri dell'ultimo strato associato all'uscita `j`. |
| `θ̄_{0,j}` | Media a priori dei parametri, inizializzata con i pesi del NNARX pre-addestrato. |
| `Λ_{0,j}` | Matrice di precisione a priori, ossia inversa della covarianza. |
| `Λ_{0,j}^{-1}` | Covarianza a priori dei parametri. |
| `n_f` | Numero di feature fornite dalla parte non lineare del NNARX. |
| `n_y` | Numero di uscite del modello. |

L'uso di un modello bayesiano consente di aggiornare sia la stima dei parametri sia l'incertezza associata.

---

## 4. Aggiornamento ricorsivo bayesiano

Per ogni campione `k` e per ciascuna uscita `j`, la media e la precisione posteriori vengono aggiornate ricorsivamente usando la nuova misura dell'impianto.

Definendo

\[
f_k^{NN}=f^{NN}(\phi_{k-1},u_{k-1};\bar\Theta),
\]

l'aggiornamento della matrice di covarianza è

\[
\Lambda_{k,j}^{-1}
=
\Lambda_{k-1,j}^{-1}
-
\frac{
\left(\Lambda_{k-1,j}^{-1}f_k^{NN}\right)
\left(\Lambda_{k-1,j}^{-1}f_k^{NN}\right)^\top
}{
\lambda_{k,j}+{f_k^{NN}}^\top\Lambda_{k-1,j}^{-1}f_k^{NN}
}.
\]

La quantità intermedia è

\[
Q_{k,j}=f_k^{NN}{w_{k,j}}^\top+Q_{k-1,j},
\]

mentre il vettore dei parametri posteriori è

\[
\bar\theta_{k,j}=\Lambda_{k,j}^{-1}Q_{k,j}.
\]

Nel testo, `w_{k,j}` rappresenta il contributo legato alla nuova osservazione dell'uscita `j`; la definizione completa non è visibile nell'estratto e deve essere recuperata dalla formulazione della regressione bayesiana adottata.

### Interpretazione

- `Λ_{k,j}^{-1}` rappresenta la covarianza posteriore dei parametri;
- `Λ_{k,j}` rappresenta la precisione posteriore;
- `θ̄_{k,j}` è la stima aggiornata dei parametri dell'ultimo strato;
- `λ_{k,j}` rappresenta un termine associato al rumore o alla precisione della misura;
- l'aggiornamento è ricorsivo e non richiede la memorizzazione dell'intero dataset storico.

Il termine sottratto nell'aggiornamento di `Λ^{-1}` è semidefinito positivo. Di conseguenza, in condizioni informative, la covarianza dei parametri tende a diminuire con l'acquisizione di nuovi dati.

---

## 5. Uscita corretta mediante BLL

Dopo l'aggiornamento bayesiano, l'uscita prevista è corretta mediante la media posteriore dei parametri:

\[
\tilde y_{k+1}
=
\bar\theta_k^\top f^{NN}(\phi_k,u_k;\bar\Theta).
\]

Per la singola uscita `j`:

\[
\tilde y_{k+1,j}
=
\bar\theta_{k,j}^\top f^{NN}(\phi_k,u_k;\bar\Theta).
\]

La correzione non è quindi un termine additivo esterno, ma una sostituzione dei pesi nominali dell'ultimo strato con la loro media posteriore aggiornata.

| Quantità | Ruolo |
|---|---|
| `y_{k+1}` | Previsione del NNARX con parametri nominali `θ_0`. |
| `ỹ_{k+1}` | Previsione corretta con parametri posteriori `θ̄_k`. |
| `θ̄_{k,j}` | Media posteriore utilizzata per l'uscita `j`. |

---

## 6. Quantificazione dell'incertezza di uscita

Per ogni uscita `j`, l'incertezza dovuta ai parametri dell'ultimo strato è quantificata da

\[
w_{h+1,j}
=
\beta_j
\sqrt{
{f^{NN}(\phi_{h|k},u_{h|k};\bar\Theta)}^\top
\Lambda_{k,j}^{-1}
f^{NN}(\phi_{h|k},u_{h|k};\bar\Theta)
}.
\]

| Simbolo | Descrizione |
|---|---|
| `w_{h+1,j}` | Semiampiezza dell'intervallo di incertezza della `j`-esima uscita prevista. |
| `β_j` | Coefficiente di scala o livello di confidenza per l'uscita `j`. |
| `Λ_{k,j}^{-1}` | Covarianza posteriore dei parametri dell'uscita `j`. |
| `f^{NN}` | Feature NNARX valutate lungo la traiettoria prevista. |

L'intervallo associato alla previsione è

\[
\tilde y_{h+1,j}\pm w_{h+1,j}.
\]

Un valore elevato di `w` indica che la traiettoria prevista attraversa una regione dello spazio delle feature per la quale i parametri sono ancora incerti. Il coefficiente `β_j` regola il compromesso tra prudenza e capacità di esplorazione.

---

## 7. Variabili del modello

### 7.1 Variabili decisionali dell'MPC

| Variabile | Tipo | Descrizione |
|---|---|---|
| `v_{h|k}` | Decisionale | Vettore dei comandi ottimizzati al passo previsto `h`, calcolati a `k`. |
| `v_{k:k+H-1|k}` | Decisionale | Sequenza completa dei comandi sull'orizzonte `H`. |
| `ζ_{h+1|k,j}` | Decisionale ausiliaria | Slack non negativa associata ai vincoli robusti sull'uscita `j`. |

La composizione fisica di `v` dipende dall'applicazione. Nel modello termico di base può includere, per esempio, le temperature di mandata della caldaia elettrica e della caldaia a gas.

### 7.2 Variabili dipendenti e previste

| Variabile | Descrizione |
|---|---|
| `y_{h+1|k}` | Uscita nominale prevista dal modello NNARX. |
| `ỹ_{h+1|k}` | Uscita corretta mediante i parametri BLL posteriori. |
| `ỹ_{h+1|k,j}` | Componente `j` dell'uscita corretta. |
| `w_{h+1|k,j}` | Semiampiezza dell'incertezza prevista per l'uscita `j`. |
| `φ_{h|k}` | Regressore NNARX lungo l'orizzonte. |
| `f^{NN}(φ,u;Θ̄)` | Vettore delle feature non lineari del NNARX. |

### 7.3 Parametri adattivi

| Variabile | Descrizione |
|---|---|
| `θ̄_{k,j}` | Media posteriore dei parametri dell'ultimo strato. |
| `Λ_{k,j}` | Precisione posteriore dei parametri. |
| `Λ_{k,j}^{-1}` | Covarianza posteriore dei parametri. |
| `Q_{k,j}` | Statistica ricorsiva impiegata nell'aggiornamento della media. |

Queste quantità vengono aggiornate prima della soluzione dell'MPC e sono trattate come parametri noti durante la singola ottimizzazione.

### 7.4 Misure e dati storici

| Variabile | Descrizione |
|---|---|
| `y_k^p` | Uscita misurata sull'impianto all'istante `k`. |
| `u_{k-τ}` | Ingresso storico applicato all'impianto. |
| `y^p_{k+1-τ}` | Uscita storica misurata. |

### 7.5 Parametri di progetto

| Parametro | Descrizione |
|---|---|
| `H` | Orizzonte di previsione dell'MPC. |
| `n_τ` | Numero di ritardi del regressore NNARX. |
| `α` | Peso complessivo del termine di esplorazione. |
| `γ_j` | Peso relativo dell'incertezza dell'uscita `j`. |
| `β_j` | Coefficiente di confidenza dell'intervallo di incertezza. |
| `c` | Soglia massima ammessa per la somma delle slack. |
| `ε` | Soglia che determina il passaggio da esplorazione a goal reaching. |
| `T_i` | Durata della fase iniziale di raccolta dati. |
| `T` | Istante finale o durata totale dell'esperimento. |
| `y̲_j`, `ȳ_j` | Limiti operativi inferiore e superiore della `j`-esima uscita. |

---

## 8. Problema MPC di esplorazione sicura

La modalità di esplorazione risolve

\[
\min_{v_{k:k+H-1|k},\,\zeta}
\sum_{h=k}^{k+H-1}
\left[
\mathcal L(v_{h|k},\tilde y_{h|k})
-\alpha\sum_{j=1}^{n_y}\gamma_j w_{h|k,j}
\right].
\]

Poiché il termine di incertezza è sottratto dal costo, la minimizzazione favorisce traiettorie con `w` elevato. Il controllore viene quindi incentivato a visitare regioni informative, mantenendo contemporaneamente il costo operativo e i vincoli di sicurezza.

### Interpretazione dei pesi

- `α = 0`: nessun incentivo all'esplorazione;
- `α > 0`: maggiore preferenza per regioni informative;
- `γ_j` alto: priorità all'esplorazione associata all'uscita `j`;
- valori molto elevati possono portare il controllore verso il bordo dei vincoli e aumentare l'uso delle slack.

---

## 9. Vincoli dell'MPC di esplorazione

I vincoli valgono per

\[
h\in[k,k+H-1],\qquad j\in[1,n_y].
\]

### 9.1 Modello nominale NNARX

\[
y_{h+1|k}
=
\theta_0^\top f^{NN}(\phi_{h|k},u_{h|k};\bar\Theta).
\]

Il vincolo propaga l'uscita nominale del modello NNARX lungo l'orizzonte.

### 9.2 Inizializzazione del regressore

\[
u_{k-\tau|k}=u_{k-\tau},
\qquad
y_{k+1-\tau|k}=y^p_{k+1-\tau},
\qquad \tau=1,\ldots,n_\tau.
\]

La memoria del NNARX è inizializzata con gli ingressi applicati e le uscite misurate dall'impianto.

### 9.3 Vincoli operativi generali

\[
\psi(v_{h|k},v_{h|k},\tilde y_{h|k})\le 0.
\]

La funzione vettoriale `ψ` raccoglie i vincoli fisici e operativi. Nell'applicazione termica può includere:

- limiti sui comandi;
- limiti sul rateo dei comandi;
- bilanci di massa ed energia;
- limiti di potenza dei generatori;
- relazioni di miscelazione;
- altri limiti di funzionamento dell'impianto.

La ripetizione di `v_{h|k}` nei primi due argomenti è quella visibile nell'estratto, ma potrebbe rappresentare due vettori diversi oppure il comando corrente e quello precedente. La definizione completa di `ψ` è necessaria per eliminare l'ambiguità.

### 9.4 Uscita corretta BLL

\[
\tilde y_{h+1|k,j}
=
\bar\theta_{k,j}^\top
f^{NN}(\phi_{h|k},u_{h|k};\bar\Theta).
\]

L'uscita utilizzata ai fini del controllo è calcolata con la media posteriore dei parametri dell'ultimo strato.

### 9.5 Incertezza dell'uscita

\[
w_{h+1|k,j}
=
\beta_j
\sqrt{
{f^{NN}}^\top\Lambda_{k,j}^{-1}f^{NN}
}.
\]

Il vincolo definisce la semiampiezza dell'intervallo di incertezza lungo la traiettoria prevista.

### 9.6 Vincoli robusti con slack

\[
\underline y_j+w_{h+1|k,j}
\le
\tilde y_{h+1|k,j}
\le
\overline y_j-w_{h+1|k,j}+\zeta_{h+1|k,j}.
\]

Il limite operativo viene ristretto mediante `w`. In assenza di slack attiva, l'intero intervallo

\[
[\tilde y-w,\,\tilde y+w]
\]

rimane entro i limiti fisici `[y̲_j, ȳ_j]`.

Nell'equazione visibile la slack compare soltanto sul limite superiore. Questa asimmetria va verificata sul testo sorgente o sull'implementazione, perché una formulazione simmetrica userebbe normalmente una slack anche sul limite inferiore oppure una slack comune su entrambi i lati.

### 9.7 Budget complessivo delle slack

\[
\sum_{h=k}^{k+H-1}\sum_{j=1}^{n_y}
\zeta_{h+1|k,j}\le c.
\]

Il parametro `c ≥ 0` limita la violazione complessiva dei vincoli robusti sull'intero orizzonte.

- `c = 0` rende i vincoli robusti hard;
- `c > 0` consente una violazione limitata per preservare la fattibilità;
- un valore troppo elevato può ridurre significativamente la protezione fornita dai vincoli robusti.

### 9.8 Non negatività delle slack

\[
\zeta_{h+1|k,j}\ge 0.
\]

La slack può soltanto rilassare il vincolo e non può restringerlo artificialmente.

---

## 10. Problema MPC di goal reaching

Quando l'incertezza è sufficientemente bassa, il termine di esplorazione e le slack vengono rimossi. Il problema diventa

\[
\min_{v_{k:k+H-1|k}}
\sum_{h=k}^{k+H-1}
\mathcal L(v_{h|k},\tilde y_{h|k})
\]

soggetto ai vincoli

\[
(23b)-(23f).
\]

La modalità di goal reaching:

- minimizza esclusivamente il costo operativo;
- mantiene la correzione bayesiana dell'uscita;
- mantiene il restringimento dei limiti mediante `w`;
- rimuove le slack, rendendo hard i vincoli robusti;
- non incentiva più il controllore a visitare regioni ad alta incertezza.

Il problema è più restrittivo rispetto alla modalità di esplorazione perché non ammette il rilassamento espresso da `ζ`.

---

## 11. Logica di commutazione

L'algoritmo prevede tre fasi.

### 11.1 Inizializzazione

Per `k = 1, ..., T_i` vengono applicati ingressi di inizializzazione, sono raccolte le misure e vengono aggiornati `θ̄_{k,j}` e `Λ_{k,j}`. La fase fornisce un primo livello di informazione al modello bayesiano prima dell'attivazione dell'MPC.

### 11.2 Esplorazione sicura

Viene risolto il problema di esplorazione. Se la somma ottima delle slack è positiva,

\[
\sum_{h=k}^{k+H-1}\sum_{j=1}^{n_y}
\zeta^*_{h+1|k,j}>0,
\]

il sistema non dispone ancora di informazione sufficiente per soddisfare i vincoli robusti senza rilassamento. Il controllore applica quindi la prima azione della sequenza esplorativa:

\[
v_k=v^*_{k|k}.
\]

### 11.3 Passaggio a goal reaching

Se la somma delle slack è nulla, viene verificato il criterio

\[
k<T_i+\varepsilon.
\]

Nell'algoritmo mostrato, quando il criterio è soddisfatto viene comunque applicata la prima azione esplorativa. In caso contrario viene risolto il problema di goal reaching e si applica la relativa prima azione ottima.

Il parametro `ε` è presentato nel testo come una tolleranza legata all'incertezza, ma nello pseudocodice visibile compare in un confronto temporale `k < T_i + ε`. Le due interpretazioni non sono perfettamente coerenti. La formulazione originale completa deve essere verificata per stabilire se `ε` sia:

- una durata aggiuntiva minima dell'esplorazione;
- una soglia sull'incertezza;
- oppure un criterio combinato.

L'algoritmo consente inoltre di tornare in esplorazione quando diventano disponibili nuovi stati informativi o aumenta nuovamente l'incertezza del modello.

---

## 12. Classificazione sintetica dei vincoli

| Categoria | Vincoli |
|---|---|
| Dinamica nominale | Equazione NNARX e propagazione del regressore. |
| Correzione bayesiana | Uscita calcolata con la media posteriore `θ̄_{k,j}`. |
| Incertezza | Definizione di `w` mediante covarianza posteriore e feature NNARX. |
| Operativi | Funzione vettoriale `ψ ≤ 0`. |
| Sicurezza robusta | Restringimento dei limiti mediante `w`. |
| Fattibilità | Slack `ζ ≥ 0` e budget complessivo `Σζ ≤ c`. |
| Commutazione | Regola basata su fase iniziale, slack e criterio `ε`. |

### Vincoli hard e soft

| Modalità | Vincoli robusti | Slack |
|---|---|---|
| Esplorazione | Soft entro il budget `c` | Attive |
| Goal reaching | Hard | Rimosse |

I vincoli fisici raccolti in `ψ` appaiono hard in entrambe le modalità, salvo diversa definizione interna della funzione.

---

## 13. Sequenza operativa completa

A ogni istante `k`, il controllore:

1. misura l'uscita reale `y_k^p`;
2. calcola le feature `f_k^{NN}` a partire dal regressore e dall'ingresso precedente;
3. aggiorna ricorsivamente, per ogni uscita `j`, `Λ_{k,j}^{-1}`, `Q_{k,j}` e `θ̄_{k,j}`;
4. costruisce le previsioni corrette `ỹ` lungo l'orizzonte;
5. calcola l'incertezza `w` lungo la traiettoria prevista;
6. risolve inizialmente l'MPC di esplorazione;
7. valuta l'uso delle slack e il criterio di commutazione;
8. mantiene la soluzione esplorativa oppure risolve il problema di goal reaching;
9. applica soltanto il primo comando ottimo;
10. incrementa `k` e ripete l'aggiornamento con la nuova misura.

---

## 14. Differenze rispetto agli altri modelli MPC

| Aspetto | MPC nominale | MPC-GP | MPC-BLL |
|---|---|---|---|
| Modello nominale | NNARX fisso | NNARX fisso | Strati interni NNARX fissi |
| Adattamento online | Assente | GP dell'errore | Ultimo strato bayesiano |
| Tipo di correzione | Nessuna | Additiva sull'uscita | Aggiornamento dei parametri di uscita |
| Memoria dati | Storia NNARX | Dataset GP mobile | Statistiche ricorsive `Λ` e `Q` |
| Incertezza | Non esplicita | Varianza GP disponibile | Covarianza dei parametri BLL |
| Uso dell'incertezza | Nessuno | Non necessariamente nei vincoli | Esplorazione e tightening dei vincoli |
| Apprendimento attivo | Assente | Assente nella formulazione base | Presente |
| Costo computazionale online | Basso | Potenzialmente elevato | Più contenuto del GP, dipendente da `n_f` e `n_y` |

---

## 15. Aspetti da chiarire per l'implementazione

Per implementare il modello in modo univoco sono necessarie le seguenti informazioni:

1. forma completa della regressione bayesiana e definizione di `w_{k,j}` nell'aggiornamento di `Q_{k,j}`;
2. significato e scelta del termine `λ_{k,j}`;
3. inizializzazione numerica di `Λ_{0,j}` e relativo livello di fiducia nel modello pre-addestrato;
4. composizione dei vettori `u`, `v`, `y` e `φ`;
5. forma esplicita del costo `L` e dei vincoli `ψ`;
6. definizione delle uscite considerate critiche e dei rispettivi limiti;
7. scelta di `α`, `γ_j`, `β_j`, `c`, `T_i` ed `ε`;
8. trattamento dell'incertezza multi-step, perché le feature future dipendono da uscite previste;
9. verifica dell'asimmetria della slack nel vincolo robusto;
10. criterio definitivo di commutazione tra esplorazione e goal reaching;
11. eventuale regolarizzazione numerica della covarianza e protezione dalla perdita di definita positività;
12. eventuale *forgetting factor* per adattarsi a dinamiche che cambiano nel tempo, non visibile nelle equazioni riportate.

---

## 16. Indicazioni per la taratura

### Matrice iniziale di covarianza

Una covarianza iniziale piccola esprime forte fiducia nel NNARX pre-addestrato e rallenta l'adattamento. Una covarianza ampia accelera l'apprendimento, ma produce intervalli `w` più grandi e può rendere inizialmente difficile la fattibilità robusta.

### Coefficiente `β_j`

Un valore elevato amplia l'intervallo di confidenza, restringe maggiormente i vincoli e rende il controllo più prudente. Lo stesso valore aumenta però l'incentivo all'esplorazione, perché `w` compare con segno negativo nella funzione obiettivo esplorativa.

### Peso `α`

`α` controlla il compromesso tra costo operativo e informatività. Una taratura eccessiva può far prevalere l'esplorazione sul raggiungimento dell'obiettivo fisico.

### Budget `c`

`c` deve essere sufficientemente grande da evitare infeasibilità durante la fase iniziale, ma abbastanza piccolo da impedire violazioni operative rilevanti. Se i limiti rappresentano vincoli realmente di sicurezza, la slack non dovrebbe rilassare i limiti fisici effettivi, ma soltanto un margine di sicurezza interno più conservativo.

---

## 17. Sintesi

Il modello MPC-BLL aggiorna online soltanto l'ultimo strato del NNARX mediante regressione lineare bayesiana. La media posteriore corregge la previsione, mentre la covarianza posteriore determina un intervallo di incertezza per ciascuna uscita.

Le variabili decisionali principali sono i comandi `v` e, nella fase di esplorazione, le slack `ζ`. Le grandezze `ỹ` e `w` sono variabili dipendenti, calcolate rispettivamente dalla media e dalla covarianza posteriori. I principali vincoli riguardano la dinamica NNARX, le condizioni operative, il restringimento robusto dei limiti di uscita e il budget delle slack.

La fase esplorativa massimizza indirettamente l'informazione raccolta sottraendo l'incertezza dal costo, ma conserva un livello di sicurezza mediante il tightening dei vincoli. Quando la conoscenza del modello è sufficiente, il controllore passa alla modalità di goal reaching e minimizza il solo costo operativo con vincoli robusti hard.
