# A new Paradigm for Dynamic Operating Envelopes and Bilateral Trading of Limits in Electricity Distribution Networks

This repository provides the implementation of Bilateral Limit exchange Under Ensured Safeness (BLUES), a novel paradigm that integrates Dynamic Operating Envelopes (DOEs) with a market-based limit exchange mechanism to enhance the flexibility and efficiency of low-voltage distribution networks.

For more detailed information, refer to the paper [here](https://).

## Table of Contents
- [Introduction](#introduction)
- [Required Data](#requireddata)
- [Usage](#usage)

## Introduction
The BLUES framework enables dynamic allocation of import/export limits, allowing customers to trade unused capacity while ensuring grid stability and efficiency. This repository contains the necessary code to reproduce the methodology presented in our paper.

## Required Data
The required data for running the code is available in the `Data` folder. The data includes:

- A Pandapower network configuration file 
- Timeseries of energy usage, both consumtpion and production


## Usage
To run the methodology, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/LimitParadigm/LimitParadigm.git
   cd LimitParadigm

2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run main notebook script:
   ```bash
   main.ipynb

***

For any questions or issues, please contact mvassallo@uliege.be.
