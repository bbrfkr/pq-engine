# Pseudo Quantum Engine
## Install
- pip (without gpu support)
    ```
    pip install git+https://github.com/bbrfkr/pq-engine
    ```

- pip (with gpu support)
    ```
    pip install git+https://github.com/bbrfkr/pq-engine[gpu]
    ```

- poetry (without gpu support)
    ```
    poetry add git+https://github.com/bbrfkr/pq-engine
    ```

- poetry (with gpu support)
    ```
    poetry add git+https://github.com/bbrfkr/pq-engine[gpu]
    ```

## Get Started
- quantum teleportation example
    ```
    python examples/quantum_teleportation.py
    ```

## Settings
### Environment Variables
- PQENGINE_USE_GPU (default: "True")
    - Whether using gpu or not. If it has empty string, gpu is not used.
- PQENGINE_ATOL (default: "1e-5")
    - atol value used by numpy or cupy.
- PQENGINE_RTOL (default: "1e-5")
    - rtol value used by numpy or cupy.
- PQENGINE_ROUNDED_DECIMAL (default: "5") 
    - approximation order of decimal used by numpy or cupy
