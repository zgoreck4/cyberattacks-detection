# Cyberattacks Detection

Projekt służy do symulacji i wykrywania cyberataków w systemie połączonych zbiorników z wykorzystaniem modeli uczenia maszynowego.

---

## Opis projektu

Celem projektu jest symulacja referencyjnego układu 4 zbiorników w stanie normalnym oraz w 4 scenariuszach cyberataków. Projekt zawiera również moduł do wykrywania cyberataków metodą bazująca na modelu odwzrorwującym działanie procesu w warunkach normalnych. System wykorzystuje modele predykcyjne szeregów czasowych (np. LSTM, MLP).

---

## Funkcjonalności

- Symulacja procesu 4 zbiorników
- Wprowadzanie scenariuszy cyberataków
- Obliczanie residuów i wykrywanie anomalii na podstawie różnych metryk (RMSE, MAE)
- Obsługa wielu modeli uczenia maszynowego (np. LSTM-MLP)
- Metody detekcji oparte na progach z możliwością konfiguracji
- Interfejs wiersza poleceń umożliwiający elastyczne uruchamianie eksperymentów

---

## Instalacja

1. Sklonuj repozytorium:

   ```bash
   git clone https://github.com/zgoreck4/cyberattacks_detection.git
   cd cyberattacks_detection
   ```
2. Utwórz i aktywuj środowisko conda:

    ```bash
    conda env create -f environment.yml
    conda activate cyberattacks_detection
    ```

## Uruchomienie
Uruchom skrypt główny z domyślnymi parametrami:
```bash
python -m cyberattacks_detection
```
Uruchom z niestandardowymi parametrami:
```bash
python -m cyberattacks_detection --save_mode --attack_scenario 1 --model_type elm
```
Aby zobaczyć wszystkie dostępne opcje:
```bash
python -m cyberattacks_detection --help
```

## Współpraca

Zapraszam do zgłaszania problemów (issues) oraz tworzenia pull requestów z ulepszeniami lub poprawkami.