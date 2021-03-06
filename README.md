# Educage_Gui
The "Educage_Gui" is a Python based GUI For Comprehensive Data Analysis & Visualization.  
The software was designed and implemented for efficient and accurate 'on the fly' data analysis.  
The following libraries were in extensive use as part of the implementeation: numpy, pandas, sklearn, matplotlib and PySimpleGUI.  
![educage](https://user-images.githubusercontent.com/83977654/125738396-c04dadb0-f1c5-44ee-9a78-c24285c02706.png)  
The "Educage" is a unique automated training platform developed in [The NeuroPlasticity lab.](https://elsc.huji.ac.il/people-directory/faculty-members/adi-mizrahi/)  
![educage](https://user-images.githubusercontent.com/83977654/125638474-76cc2aa7-50bc-4027-8dd0-14d97fd2bb00.png)

The "Educage" allows efficient training of several mice simultaneously.  
Here, we used the system to train mice to discriminate among pure tones or complex sounds.  
For more inforamtion about the reaserch check the original paper:  [Maor et al.(2020). Neural correlates of learning pure tones or natural sounds in the auditory cortex.](https://www.frontiersin.org/articles/10.3389/fncir.2019.00082/full)  
  
The Educage_Gui main functionality:  
  
**Data Handling:**  
- Efficient data set construction from the "Educage" raw log files.  
- Loading automatically previous data sets for efficient Ongoing analysis  
- Export data sets and figures.  

  
**Analysis:**  
- Behavior rates (hit, miss, false alarm, correct rejection) with adjustable bin size slider - selecting multiple mice for automatically mean and standard deviation calculation  
![image](https://user-images.githubusercontent.com/83977654/125960881-d1aff91a-27c1-4c51-bc90-9df2578fe94c.png)  
- 2D Behavior analysis - investigate positive / negative sequences  
![image](https://user-images.githubusercontent.com/83977654/125958545-2e837b1d-329e-4010-a909-11b64dac1c46.png)
- Positive / Negative sequences histogram  
![image](https://user-images.githubusercontent.com/83977654/125959767-d2dac492-478d-44af-9bbb-d60a61e971f6.png)  
- [Markov decision process](https://en.wikipedia.org/wiki/Markov_decision_process) with adjustable slider for specific chain order  
![image](https://user-images.githubusercontent.com/83977654/125960514-0aa8dafb-cfc7-4f65-9d04-7b05c04b56e2.png)

- [Sensitivity index](https://en.wikipedia.org/wiki/Sensitivity_index) - d'  
![image](https://user-images.githubusercontent.com/83977654/126001583-293302ce-9926-4842-af0e-f24b107f6409.png)

-Behavior pie  
![image](https://user-images.githubusercontent.com/83977654/125958120-f34e1783-ecb1-4ebe-8987-673133ac09d8.png)

 
