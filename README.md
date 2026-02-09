# Car-Anti-Theft-with-Facial-Recognition
Car anti-theft system designed for Raspberry Pi with in-cabin camera based facial recognition and telegram-based live alerts. The anti-theft system also includes a fail-safe door lock, electronic immobilizer and a sound alarm system.

## 📄 Related Publication

This repository accompanies the paper:

**Car Anti-theft System Using Driver Facial Biometrics Authentication and Telegram Alert**  
Sourav Kumar, R. Karthika, 2026  
Springer, Power Engineering and Intelligent Systems (PEIS 2025)  
DOI: https://doi.org/10.1007/978-981-96-9724-3_6

## You can download Rasperry Pi Image from here - https://github.com/Qengineering/RPi-Bullseye-DNN-image

## Installation

Before running the project, install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Demo 
  ```bash
  python CarLock_FaceApp.py
  ``` 
> **Note:** While running on Raspberry Pi, comment out the commented lines. Place all trusted users images in KnownUser directory. Replace your created Telegram Bot token ID in Configuration class. For more details on creating and using your own Telegram Bot, visit: https://core.telegram.org/bots/features#botfather.

Don't forget to star the repo if it is helpful for your research 

## Results
<table>
  <tr>
    <td align="center">
      <img src="assets/Real-Time_Hardware.png" width="300"><br>
      <em>Figure 1: Hardware Setup</em>
    </td>
    <td align="center">
      <img src="assets/Telegram_Alerts.svg" width="500"><br>
      <em>Figure 2: Telegram Alerts</em>
    </td>
  </tr>
</table>

## Reference 
* https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch - For Facial Recoginition

## Citation

If you use this work, please cite:

```bibtex
@InProceedings{10.1007/978-981-96-9724-3_6,
author="Kumar, Sourav
and Karthika, R.",
editor="Shrivastava, Vivek
and Bansal, Jagdish Chand
and Panigrahi, Bijaya Ketan",
title="Car Anti-theft System Using Driver Facial Biometrics Authentication and Telegram Alert",
booktitle="Power Engineering and Intelligent Systems",
year="2026",
publisher="Springer Nature Singapore",
address="Singapore",
pages="75--89",
isbn="978-981-96-9724-3"
}
```


