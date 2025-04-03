<p align="center">
    <img src="assets/logo.png" width="300">
</p>

## Causal inertia proximal Mamba network for magnetic resonance image reconstruction



> **Abstract:**  Accurate and rapid Magnetic Resonance Imaging (MRI) is critical for clinical diagnosis. However, different sampling strategies and datasets act as confounding factors, significantly impacting the quality of image reconstruction. While existing methods can capture correlations between data during the imaging process, they overlook the deeper associations rooted in causal relationships. To address this issue, this paper proposes a Causal Inertial Proximal Mamba Network (CIPM-Net) to achieve robust and efficient MRI reconstruction. Specifically, we present a causal inertial proximal iterative algorithm that eliminates biases caused by confounding factors using a causal model, improving the ability of the algorithm to identify spurious correlations. Furthermore, to achieve an effective balance between global perception and computational efficiency during the reconstruction process, the proposed algorithm is extended into a Mamba-based network. At the channel level, a Causal Channel Mamba (CCM) module is introduced to suppress irrelevant channel features, thereby enhancing the quality of the reconstructed images. For spatial features, a novel Causal Spatial Mamba (CSM) module is designed to adaptively assign varying weights to pixel points, optimizing the extraction of spatial information and improving overall reconstruction performance. Additionally, to account for causal relationships in the frequency domain, a Causal Frequency Mamba (CFM) module is introduced to effectively capture complex and elongated pathological features. Extensive experiments across varying acceleration factors demonstrate the superiority of the proposed approach, with results on the IXI and in-house clinical datasets showing that CIPM-Net outperforms baseline methods by an average of 5.75/6.22 and 7.13/7.38 in PSNR and SSIM, respectively.

<p align="center">
    <img src="CIPM.png" style="border-radius: 15px">
</p>
![CIPM.png](CIPM.png)

