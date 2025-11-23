import torch
def is_gpu_supported_by_pytorch13():
    """
    Check if the GPU supports PyTorch 1.3.0 based on compute capability.
    PyTorch 1.3.0 (with CUDA 10.1) supports compute capabilities up to 7.5 (Turing architecture).
    GPUs with compute capability ≥ 8.0 (Ampere+) are NOT supported.
    """
    if not torch.cuda.is_available():
        return False, "No CUDA-enabled GPU detected"
    
    try:
        # Get GPU compute capability
        major, minor = torch.cuda.get_device_capability(0)
        compute_capability = major + minor / 10
        
        # PyTorch 1.3.0 supports up to compute capability 7.5
        if compute_capability <= 7.5:
            return True, f"GPU with compute capability {compute_capability} is supported by PyTorch 1.3.0"
        else:
            return False, f"GPU with compute capability {compute_capability} is NOT supported by PyTorch 1.3.0 (requires <=7.5)"
    
    except Exception as e:
        return False, f"Error checking GPU compatibility: {str(e)}"


# Usage example
if __name__ == "__main__":

    
    supported, message = is_gpu_supported_by_pytorch13()
    print(supported)
    print(message)
    if not supported:
        print("Recommendation: Upgrade PyTorch to 1.7.0+ for better GPU support")
    