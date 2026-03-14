# Task Per Un Agente AI: Prior Debole Su `a` E `b`

## Obiettivo
Implementare una nuova regolarizzazione debole sui valori medi di `a` e `b`, senza cambiare il resto della fisica del modello.

Lo scopo e' migliorare l'identificabilita' dei parametri di attrito assoluti, mantenendo invariato il vincolo gia' esistente su `a-b`.

## Contesto tecnico
Nel repository:
- `a`, `b`, `D_c` sono prodotti da `FrictionNetwork` in `model.py`
- oggi esiste gia' una regolarizzazione su `a-b` in `loss.py`
- il training compone loss dati + loss fisiche in `main.py`

Problema attuale:
- il modello controlla il range di `a` e `b`
- controlla parzialmente `a-b`
- ma lascia troppo spazio a compensazioni tra `a`, `b`, `theta` e `D_c`

Questa task NON deve introdurre:
- prior su `D_c`
- annealing aggiuntivo
- staged training
- modifiche ai range di output in `model.py`

## Modifica da implementare
Aggiungere una nuova loss:

```python
((a.mean() - a_prior_mean) / a_prior_std).square()
+ ((b.mean() - b_prior_mean) / b_prior_std).square()
```

Questa loss deve essere debole, configurabile e separata dalla loss gia' esistente su `a-b`.

## File da modificare

### 1. `loss.py`
Aggiungere:
- una funzione `compute_L_ab_prior(...)`
- un metodo `ab_prior(...)` dentro `PINNLoss`

Dettagli:
- input minimi: `a`, `b`, `a_prior_mean`, `b_prior_mean`, `a_prior_std`, `b_prior_std`
- usare direttamente `a.mean()` e `b.mean()`
- assicurarsi che `a_prior_std` e `b_prior_std` non portino a divisioni instabili

Output atteso:
- una loss scalare PyTorch

### 2. `config.py`
Aggiungere nuove chiavi a `DEFAULT_CONFIG`:
- `lambda_ab_prior`
- `a_prior_mean`
- `b_prior_mean`
- `a_prior_std`
- `b_prior_std`

Valori iniziali consigliati:
```python
"lambda_ab_prior": 0.0,
"a_prior_mean": 0.015,
"b_prior_mean": 0.020,
"a_prior_std": 0.010,
"b_prior_std": 0.010,
```

Nota:
- `lambda_ab_prior` deve essere `0.0` di default per non alterare il comportamento attuale se non richiesto

### 3. `main.py`
Integrare la nuova loss nel training.

Richieste:
- calcolare `L_ab_prior` durante il loop fisico/collocation
- aggiungerla a `L_total` con peso `lambda_ab_prior`
- salvarla nel `history`

Nel log/history aggiungere anche:
- `L_ab_prior`
- `a_mean`
- `a_std`
- `b_mean`
- `b_std`

Vincoli:
- non rimuovere o modificare la loss attuale `L_friction_reg`
- non cambiare la composizione delle altre loss salvo aggiungere questo termine

## Comportamento richiesto

### Caso 1: default config
Con:
```python
"lambda_ab_prior": 0.0
```
il training deve restare equivalente al comportamento precedente.

### Caso 2: prior attivo
Con:
```python
"lambda_ab_prior" > 0
```
il modello deve ricevere il nuovo vincolo su `mean(a)` e `mean(b)`.

## Verifiche minime da fare

### Verifica tecnica
- `python -m py_compile loss.py main.py config.py`

### Verifica funzionale
Fare due run comparabili:

1. baseline:
```bash
python main.py --device cuda --checkpoint checkpoints/baseline_ab.pt
```

2. prior attivo:
```bash
python main.py --device cuda --checkpoint checkpoints/ab_prior.pt --config-json '{"lambda_ab_prior":0.02,"a_prior_mean":0.015,"b_prior_mean":0.020,"a_prior_std":0.010,"b_prior_std":0.010}'
```

Poi confrontare:
- `history.csv`
- `eval.csv`
- diagnostica:
  - mappe `a`
  - mappe `b`
  - mappa `a-b`
  - `05_tau_scatter.png`
  - `06_aging_residual.png`

## Criteri di accettazione
- il codice compila
- il training parte con e senza prior
- con `lambda_ab_prior=0` non ci sono regressioni funzionali
- `history.csv` contiene le nuove colonne
- il termine `L_ab_prior` compare nel logging
- la nuova loss e' separata da `L_friction_reg`

## Note progettuali
- Non cercare di "migliorare tutto" in questa task
- Non introdurre prior su `D_c` adesso
- Non aggiungere freeze/unfreeze o training multi-stage
- Non modificare il significato fisico delle loss esistenti

## Output atteso dall'agente
L'agente deve consegnare:
- patch ai file richiesti
- breve summary di cosa ha cambiato
- conferma di compilazione
- eventuale comando di test consigliato
