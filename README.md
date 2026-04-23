# Ransomware Detection Using Machine Learning

A Python-based ransomware detection system built on a **Random Forest classifier**, trained on the [Zenodo 2024 Ransomware Dataset](https://zenodo.org/records/13890887) — a labeled collection of over 21,000 samples spanning 26 malware families, including WannaCry, REvil, LockBit, and Ryuk.

## Overview
Traditional signature-based antivirus tools struggle against modern ransomware that uses obfuscation and polymorphic code to evade detection. This project takes a machine learning approach, training a Random Forest classifier on real-world ransomware data to identify malicious behavior patterns without relying on known signatures.

## Features
- 99%+ classification accuracy on the Zenodo 2024 dataset
- Covers 26 ransomware families including WannaCry, REvil, LockBit, and Ryuk
- Outputs an HTML report with accuracy, precision, recall, F1-score, and confusion matrix
- Reproducible results via fixed random seed

## Tech Stack
- Python
- scikit-learn
- pandas

## Dataset
[Zenodo 2024 Ransomware Dataset](https://zenodo.org/records/13890887) — not included in this repo due to size. Download the CSV and place it in the project root before running.

## How to Run
1. Clone the repo:
