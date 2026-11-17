# Event-Triggered Asynchronous Federated Learning for Spiking Neural Networks (ETA-FL)

## 1. Project Overview

This project implements an Event-Triggered Asynchronous Federated Learning (ETA-FL) framework for Spiking Neural Networks (SNNs). The framework aims to reduce communication overhead during federated training while maintaining classification performance.

The system uses a Standard Leaky Integrate-and-Fire (LIF) SNN model and the N-MNIST event-based dataset. Multiple edge clients train the model locally, while a central server aggregates the received model updates.

The framework investigates:
- Synchronous Federated Averaging (FedAvg).
- Fixed-threshold event-triggered communication.
- Adaptive-threshold event-triggered communication.
- Asynchronous model aggregation.
- Performance under IID and non-IID data distributions.
- Classification and communication performance.

## 2. Technologies Used

- Python
- PyTorch
- Flower Federated Learning
- snnTorch
- NumPy
- Pandas
- Matplotlib
- CUDA (GPU acceleration, when available)


## 3. Dataset

The project uses the N-MNIST (Neuromorphic-MNIST) event-based dataset.

## 4. Federated Learning Strategies

The following strategies are investigated:

1. **FedAvg:** Synchronous aggregation of client model updates.
2. **Fixed Threshold:** Clients transmit updates when the fixed triggering condition is satisfied.
3. **Adaptive Threshold:** Communication is controlled using an adaptive triggering threshold.
4. **Asynchronous Aggregation:** The server processes client updates as they arrive without waiting for all clients.

## 5 Experimental Results

The proposed ETA-FL framework was evaluated using the N-MNIST dataset to investigate classification performance, communication efficiency, and the effect of different client data distributions.

# 5.1 Overall Findings

Standard LIF provides the best classification performance among the evaluated SNN variants.

Fixed and adaptive event-triggering mechanisms reduce communication requirements.

Adaptive triggering achieves up to 96% communication reduction in the evaluated five-client experiment.

Asynchronous aggregation allows client updates to be processed without waiting for all clients.

Severe non-IID data significantly affects classification performance.

Further experiments will investigate scalability across larger client configurations and computational efficiency.