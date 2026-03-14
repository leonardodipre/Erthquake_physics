# Contesto Rapido Per Un Agente AI

## Cosa fa questo progetto
Questo repository allena un modello `HybridPINN` per ricostruire l'evoluzione dello slip su una faglia e dei parametri di attrito a partire da serie temporali GNSS e matrici di Green elastiche. L'obiettivo non e' solo fittare i dati di superficie, ma ottenere una soluzione coerente anche con la fisica rate-and-state friction.

## Dati in ingresso
- `dataset_scremato/`: serie temporali GNSS per stazione in CSV.
- Colonne tipiche: `date`, `E`, `N`, `U` e spesso anche `E_clean`, `N_clean`, `U_clean`.
- Le componenti sono spostamenti in metri; il loader preferisce le colonne `*_clean`.
- `green_out/`: operatori elastici e geometria della faglia.
- File chiave:
  - `K_cd_disp.npy`: da slip di faglia a spostamento superficiale.
  - `K_ij_tau.npy`: interazione elastica tra patch di faglia.
  - `station_ids.npy`: ordine delle stazioni usato ovunque.
  - `fault_mesh.npz` e `green_summary.json`: mesh e metadati della faglia.

## Come vengono usati i dati
Il loader (`dataset.py`) legge i CSV GNSS, allinea le stazioni all'ordine di `station_ids.npy`, costruisce matrici dense di osservazioni e maschere di validita', e stima anche le velocita' superficiali locali da finestre temporali. Applica filtri robusti, esclusione di stazioni problematiche e campiona:
- tempi dati, usati per confrontare modello e GNSS;
- tempi di collocazione fisica, usati per imporre i vincoli PINN.

## Modello
Il cuore e' `HybridPINN` in `model.py`.
- `SlipNetwork(xi, eta, t) -> s, theta`
- `FrictionNetwork(xi, eta) -> a, b, D_c`

Da queste quantita' il modello calcola:
- `V = ds/dt`
- `u_surface = K_cd @ s`
- `v_surface = K_cd @ V`
- `tau_elastic = tau_0 + tau_dot * t - K_ij @ s`
- `tau_rsf` dalla legge rate-and-state

La faglia e' discretizzata in patch normalizzate in coordinate `xi, eta`. Nel setup attuale la mesh in `green_out` ha `87 x 22 = 1914` patch.

## Loss e training
`main.py` allena il modello combinando:
- fit ai dati GNSS di posizione;
- fit alle velocita' GNSS;
- coerenza tra `tau_elastic` e `tau_rsf`;
- legge di aging per `theta`;
- smoothness spaziale dello slip;
- regolarizzazione di `(a-b)` e continuita' temporale di `V`.

Output standard del training:
- checkpoint `.pt`
- cronologia `.history.csv`

Comandi tipici:
```bash
python main.py --device cuda --steps 40000 --checkpoint checkpoints/run_2000_2008.pt
python eval.py --device cpu --checkpoint checkpoints/run_2000_2008.pt --out-csv checkpoints/eval.csv
python diagnose.py --checkpoint checkpoints/run_2000_2008.pt
```

## Output finali
- `checkpoints/*.pt`: pesi del modello + config.
- `checkpoints/*.history.csv`: andamento delle loss e statistiche di training.
- `eval.py`: CSV con RMSE/MAE su ogni tempo osservato.
- `predict.py`: snapshot numerici (`.npz`) con campi `s`, `V`, `theta`, `a`, `b`, `D_c`, `tau_elastic`, `tau_rsf`, `u_surface`, `v_surface`.
- `diagnose.py`: immagini diagnostiche e ranking di sensitivita' delle stazioni.
- `we-app/scripts/export_web_data.py`: esporta JSON per la web app a partire da checkpoint, evaluation e diagnostica.

## Regole pratiche per un altro agente
- Non cambiare mai l'ordine delle stazioni o delle patch senza aggiornare anche gli operatori di Green.
- Prima di modificare il modello, controlla sempre `green_summary.json` e `station_ids.npy`.
- Se i risultati peggiorano, guarda prima `eval.csv`, poi `history.csv`, poi le mappe diagnostiche.
- Se fai esperimenti, salva sempre checkpoint, CSV di eval e diagnostica nello stesso folder/versione.
