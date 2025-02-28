# Thesis - Identification and Classification of cyber attack in Automotive

## 1. PHASE:

Train&test Random Forest on each of the following 4 datasets**:

![Screenshot 2025-01-18 alle 4 15 06 PM](https://github.com/user-attachments/assets/fdd8b96c-0f9a-45d1-86af-4aa869bb0a13)

**.csv file download here:  
www.dropbox.com/scl/fo/9rwsf9pclhvv9xxloojom/AF7JeRW893grZkigkulkAHk?rlkey=gglzjap922q57acw8vfp2almh&e=1&st=b7r7855u&dl=0

#### DATASET CHARATERISTICS
HCRL provideS car-hacking datasets which include DoS attack, fuzzy attack, spoofing the drive gear, and spoofing the RPM gauge. Datasets were constructed by logging CAN traffic via the OBD-II port from a 
real vehicle while message injection attacks were performing. Datasets contain each 300 intrusions of message injection. Each intrusion performed for 3 to 5 seconds, and each dataset has total 30 to 40 
minutes of the CAN traffic.

1.    **DoS Attack** : Injecting messages of ‘0000’ CAN ID every 0.3 milliseconds. ‘0000’ is the most dominant.
2.    **Fuzzy Attack** : Injecting messages of totally random CAN ID and DATA values every 0.5 milliseconds.
3.    **Spoofing Attack (RPM/gear)** : Injecting messages of certain CAN ID related to RPM/gear information every 1 millisecond.


#### DATA ATTRIBUTES
1.    **Timestamp** : recorded time (s)
2.    **CAN ID** : identifier of CAN message in HEX (ex. 043f)
3.    **DLC** : number of data bytes, from 0 to 8
4.    **DATA[0~7]** : data value (byte)
5.    **Flag** : T or R, T represents injected message while R represents normal message


#### RESULTS OF THE CLASSIFICATIONS

![Screenshot 2025-01-18 alle 5 56 08 PM](https://github.com/user-attachments/assets/0a2b7158-7071-4a35-8ff7-9b0f47088d4f)




## 2. PHASE:
   
1. Concatenate the datasets (concat.py file) with label update:


   0-> normal run  

   1-> Dos attack

   2-> Fuzzy attack

   3-> Spoofing RPM attack

   4-> Spoofing GEAR attack
   

![Screenshot 2025-01-18 alle 5 56 30 PM](https://github.com/user-attachments/assets/cf45c997-dbbe-43c1-919e-63dd6281f1e6)



2. Build an IDS (Intrusion detection System) based on a multi-class classification with Random Forest. (IDS_multiclass.py)




## 3.PHASE

Results for the multi-class classification:

![Screenshot 2025-01-18 alle 5 56 57 PM](https://github.com/user-attachments/assets/be9f13f3-9d0b-4b6c-af81-7014b8bb9c6d)


