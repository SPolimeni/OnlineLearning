# Checklist di pubblicazione e handoff del repository

## 1) Release base
- [ ] Aggiungere una licenza open-source corretta, es. MIT / Apache 2.0
- [ ] Creare una documentazione pubblica minima:
  - overview del progetto
  - installazione
  - prerequisiti
  - architettura
  - workflow di esecuzione
  - limitazioni note
- [ ] Verificare che il README non contenga riferimenti troppo specifici al setup locale
- [ ] Separare chiaramente:
  - codice sorgente
  - dati / modelli / esempi
  - artefatti generati
  - file di utility e supporto

## 2) Configurazione e path
- [ ] Centralizzare la configurazione in un unico punto
- [ ] Usare un resolver di path robusto, basato su root del repo o config
- [ ] Rimuovere dipendenze da cartelle locali non standard dove possibile
- [ ] Aggiungere controlli iniziali:
  - file obbligatori presenti
  - modello caricato
  - dati disponibili
  - solver / dipendenze accessibili
- [ ] Documentare chiaramente i casi live vs debug vs test

## 3) Stabilità del codice
- [ ] Convalidare sempre le dimensioni dei vettori di stato/ingresso/uscita
- [ ] Inserire guard check per:
  - bounds dei parametri
  - vincoli di ottimizzazione
  - dimensione del problema
  - presenza di dati mancanti
- [ ] Verificare che il solver abbia fallback chiari ed errori leggibili
- [ ] Evitare logica nascosta in script di esecuzione troppo “magica”

## 4) Test e validazione
- [ ] Aggiungere una smoke test minima:
  - import dei moduli principali
  - compilazione Python dei file critici
  - caricamento configurazione
  - validazione di un path di esempio
- [ ] Aggiungere test per:
  - parsing dei dati
  - path resolution
  - validazione input/metadata
  - comportamento di fallback del solver
- [ ] Eseguire un controllo di regressione sulla sintassi dei file principali

## 5) Ambiente e riproducibilità
- [ ] Creare un ambiente pinato con dipendenze note
- [ ] Includere i prerequisiti minimi per:
  - CasADi
  - PyTorch / GPyTorch / altre dipendenze modello
  - FMU / OPC UA / solver
- [ ] Documentare chiaramente i requisiti di sistema
- [ ] Specificare quali componenti sono obbligatori e quali sono opzionali

## 6) Repository hygiene
- [ ] Escludere artefatti generati da tracking
- [ ] Fare pulizia finale di file legacy o duplicati
- [ ] Mantenere una sola “fonte di verità” per dati di esempio e modelli
- [ ] Verificare che i file root non contengano dati di runtime non organizzati
- [ ] Separare i risultati sperimentali dagli asset di progetto

## 7) Documentazione operativa
- [ ] Scrivere una quickstart chiara in 5–10 minuti
- [ ] Scrivere una sezione “Known limitations”
- [ ] Documentare:
  - come avviare il run live
  - come usare la modalità debug
  - come interpretare output / log
  - come riconoscere failure del solver o dati invalidi
- [ ] Indicare esplicitamente dove le assunzioni di impianto sono forti

## 8) Pubblicazione / handoff finale
- [ ] Verificare che la repo sia presentabile a un collega senza il setup originale
- [ ] Aggiungere un documento di onboarding per nuovo maintainer
- [ ] Aggiungere un file di changelog minimo
- [ ] Preparare una nota “what this project is / is not”
- [ ] Fare una verifica finale: build / import / compile / smoke test

## Priorità realistica
Se si vuole fare il passaggio minimo ma utile, le prime 4 cose da fare sono:

1. Licenza
2. Quickstart + prerequisiti
3. Config/path centralizzati
4. Smoke test / py_compile / import check

Queste danno il maggiore beneficio senza rischiare di cambiare la logica di controllo.
