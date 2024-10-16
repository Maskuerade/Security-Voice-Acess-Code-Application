# Security Voice-code Access

## Overview
Security Voice-code Access is a software application designed to enhance security using voice recognition technology. The system operates in two distinct modes, allowing for flexible access control based on voice commands and individual voice fingerprints.

## Features

### Mode 1: Security Voice Code
- Access is granted only upon the correct verbal pass-code.
- Valid pass-codes include:
  - "Open middle door"
  - "Unlock the gate"
  - "Grant me access"
- Custom sentences can be configured as long as they do not share similar words.

### Mode 2: Security Voice Fingerprint
- Grants access based on voice recognition for specific individuals.
- The software can recognize and allow access to one or more of the original eight users who provide the valid pass-code sentence.

## User Interface Components
The user interface provides an intuitive experience with the following features:
- **Record Voice-code Button**: Start recording the spoken pass-code.
- **Spectrogram Viewer**: Visual representation of the spoken voice-code.
- **Analysis Summary**:
  - **Pass-code Match Table**: Displays how closely the spoken sentence matches each of the three pass-code sentences.
  - **Voice Match Table**: Shows how closely the spoken voice matches the stored voice samples of the eight individuals.
- **Access Result Indicator**: Displays "Access gained" or "Access denied" based on the analysis results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/security-voicecode-access.git
   cd security-voicecode-access
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Select the desired operation mode (Voice Code or Voice Fingerprint).
2. Click the "Record Voice-code" button to capture your voice input.
3. Review the spectrogram and analysis summary.
4. Check the access result to see if you have been granted access.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Voice recognition libraries and frameworks that enable this project.
- The inspiration and guidance from security research in voice authentication.


## Contributors

<table>
  <tr>
    <td align="center">
    <a href="https://github.com/Youssef-Ashraf71" target="_black">
    <img src="https://avatars.githubusercontent.com/u/83988379?v=4" width="150px;" alt="Youssef Ashraf"/>
    <br />
    <sub><b>Youssef Ashraf</b></sub></a>
    </td>
    <td align="center">
    <a href="https://github.com/mouradmagdy" target="_black">
    <img src="https://avatars.githubusercontent.com/u/89527761?v=4" width="150px;" alt="Mourad Magdy"/>
    <br />
    <sub><b>Mourad Magdy</b></sub></a>
    <td align="center">
    <a href="https://github.com/ZiadMeligy" target="_black">
    <img src="https://avatars.githubusercontent.com/u/89343979?v=4" width="150px;" alt="Ziad Meligy"/>
    <br />
    <sub><b>Ziad Meligy</b></sub></a>
    </td>
    </td>
    <td align="center">
    <a href="https://github.com/Maskuerade" target="_black">
    <img src="https://avatars.githubusercontent.com/u/106713214?v=4" width="150px;" alt="Mariam Ahmed"/>
    <br />
    <sub><b>Mariam Ahmed</b></sub></a>
    </td>
      </tr>
 </table>



## Contact
For questions or suggestions, please contact the project maintainer at [your-email@example.com]. 

---
