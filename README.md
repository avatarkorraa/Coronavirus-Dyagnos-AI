# Coronavirus Dyagnos-AI

## Descrizione del Progetto

"Coronavirus Dyagnos-AI" è un progetto di intelligenza artificiale focalizzato sullo sviluppo di un approccio migliore rispetto a quello di uno studio scelto già pubblicato in letteratura per risolvere un task di ML. Esso si basa sull'utilizzo del Machine Learning per diagnosticare malattie come l'influenza H1N1 e il COVID-19, con un'enfasi particolare sull'explainability del modello. Il progetto mira a migliorare la precisione e l'affidabilità delle diagnosi utilizzando dati clinici e approcci di explainability per rendere comprensibile il processo decisionale dei modelli.

## Struttura del Progetto

La documentazione del progetto è strutturata nel seguente modo:

### 1. Introduzione
Il progetto inizia con un'analisi del problema del Machine Learning nella medicina di precisione, affrontando temi come l'applicazione dei modelli IA al COVID-19, la sfida dell'explainability e la gestione dei dati mancanti.

### 2. Data Understanding and Data Preparation
- **Data Gathering**: I dati utilizzati sono stati raccolti da studi scientifici disponibili in letteratura.
- **Data Examination e Data Cleaning**: Esame e pulizia dei dati per garantire la qualità del dataset.
- **Data Exploration**: Esplorazione e analisi preliminare dei dati.
- **Data Splitting**: Suddivisione del dataset per il training e la valutazione dei modelli.

### 3. Sviluppo del Modello
- **Scelta del Classificatore**: Analisi dei vari classificatori possibili e scelta di quelli più adatti.
- **Implementazione**: Dettagli sull'implementazione dei modelli, inclusi Decision Tree e Random Forest.
- **Explainability**: Tecniche utilizzate per migliorare la comprensione dei modelli da parte degli utenti finali.

### 4. Training e Valutazione
- **Metriche di Valutazione**: Metodi utilizzati per valutare le performance dei modelli.
- **Valutazione dei Modelli**: Dettagli sulla performance dei modelli implementati, con particolare attenzione a Decision Tree e Random Forest.

### 5. Explainability dei Modelli Implementati
- **Feature Importance e SHAP**: Approcci per spiegare le predizioni dei modelli.
- **Prototipo di Interfaccia**: Sviluppo di un'interfaccia grafica per rendere le predizioni più accessibili e comprensibili.

### 6. Conclusioni
- **Problematiche e Soluzioni**: Analisi delle principali sfide incontrate e delle soluzioni adottate.
- **Risultati Finali**: Sintesi dei risultati ottenuti dal progetto.

La repository contiene anche l'implementazione dei modelli sviluppati per il progetto.

## Installazione

Per utilizzare questo progetto, segui i passaggi seguenti:

1. **Clona il repository**:
   ```bash
     git clone https://github.com/avatarkorraa/Coronavirus-Dyagnos-AI.git

2. **Installa le dipendenze**:
     ```bash
     pip install -r requirements.txt

## Riferimenti
Il materiale e il dataset utilizzato per questo progetto appartengono ad uno studio reperibile al seguente link:

[Using machine learning of clinical data to diagnose COVID-19: a systematic review and meta-analysis](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-01266-z)
