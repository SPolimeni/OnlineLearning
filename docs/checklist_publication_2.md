# Checklist pubblicazione - attività ancora aperte

## 1. Licenza e pubblicazione

- [ ] Aggiungere una licenza open source approvata, ad esempio MIT o Apache 2.0.
- [ ] Verificare e documentare i diritti di redistribuzione di modelli, FMU, dati e risultati generati.
- [ ] Aggiungere metadati di citazione, autori, DOI e contatto del maintainer.

## 2. Configurazione e robustezza

- [x] Completare la centralizzazione della configurazione, rimuovendo i parametri ancora hard-coded.
- [x] Aggiungere controlli iniziali espliciti per:
  - [x] file richiesti;
  - [x] modelli caricati;
  - [x] dati disponibili;
  - [x] solver e dipendenze accessibili.
- [ ] Aggiungere la validazione delle dimensioni di stati, ingressi e uscite.
- [ ] Aggiungere guard check per limiti dei parametri, dimensione del problema e dati mancanti.
- [ ] Documentare in modo univoco i casi live, debug e test.

## 3. Test e validazione

- [ ] Estendere lo smoke test per includere caricamento della configurazione e validazione di un percorso di esempio.
- [ ] Aggiungere test per il parsing dei dati.
- [ ] Aggiungere test per la validazione di input e metadati.
- [ ] Aggiungere test per il comportamento di fallback del solver.
- [ ] Correggere il test sugli identificativi sintetici della configurazione InfluxDB.
- [ ] Uniformare il messaggio di errore della modalità live al test previsto, oppure aggiornare il test con il comportamento approvato.
- [ ] Eseguire e mantenere una validazione finale automatizzata di import, compilazione, smoke test e test unitari.
- [ ] Validare il percorso di simulazione FMU e registrare un baseline numerico approvato.

## 4. Ambiente e riproducibilità

- [ ] Rendere completamente riproducibile l’ambiente: fissare le versioni ancora non esatte e aggiungere un lockfile o un export verificabile.
- [ ] Registrare configurazione, condizioni iniziali, versioni, hash degli asset, sequenza degli stati del solver e tolleranze approvate.
- [ ] Definire una baseline numerica per traiettorie, costi, primi comandi e confronti tra strategie.

## 5. Igiene del repository e asset

- [ ] Completare la pulizia di script e documentazione legacy o duplicati.
- [ ] Definire una singola fonte di verità per dati di esempio, modelli e FMU.
- [ ] Aggiungere un manifest degli asset con ruolo, provenienza, hash e stato di redistribuzione.
- [ ] Verificare che dati runtime, risultati sperimentali e asset di progetto siano separati chiaramente.
- [ ] Confermare che gli artefatti generati restino esclusi dal repository pubblico.

## 6. Documentazione operativa

- [ ] Aggiungere una procedura operativa per un’esecuzione live autorizzata, separata dal percorso pubblico disabilitato.
- [ ] Documentare il significato degli output, dei file generati e dei log.
- [ ] Documentare come riconoscere errori del solver, dati non validi e fallback.
- [ ] Identificare esplicitamente le assunzioni forti relative all’impianto e al modello.

## 7. Handoff finale

- [ ] Aggiungere un documento di onboarding per un nuovo maintainer.
- [ ] Aggiungere un changelog minimo.
- [ ] Aggiungere una nota esplicita su cosa è e cosa non è il progetto.
- [ ] Verificare che il repository sia utilizzabile da un collega senza l’ambiente originale.
- [ ] Eseguire il controllo finale completo prima della pubblicazione.
