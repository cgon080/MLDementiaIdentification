#Pre-processing data for ML 
#Cristian Gonzalez-Prieto

#Libraries to use
library(lubridate)
library(dplyr)
library(ggplot2)
library(readxl)
library(tidyverse)
library(mice)
library(VennDiagram)
library(fastDummies)
library(tidyr)

#Directory
setwd("C:/Users/cgon080/OneDrive - The University of Auckland/Documents/ML")

#1. Import data
soc = read.csv("Full data/master_deidentified.csv", header = TRUE, sep = ",")
IPA = read.csv("Full data/INP_deidentified.csv", header = TRUE, sep = ",")
EDA = read.csv("Full data/EDA_deidentified.csv", header = TRUE, sep = ",")
OPA = read.csv("Full data/OPA_deidentified.csv", header = TRUE, sep = ",")
cont = read.csv("Full data/contact_deidentified.csv", header = TRUE, sep = ",")
CAM = read.csv("Full data/cam_deidentified.csv", header = TRUE, sep = ",")
inter = read.csv("Full data/interRAI_deidentified.csv", header = TRUE, sep = ",")
lab = read.csv("Full data/lab_deidentified.csv", header = TRUE, sep = ",")
Med = read.csv("Full data/med_deidentified.csv", header = TRUE, sep = ",")
arc = read.csv("Full data/arc_deidentified.csv", header = TRUE, sep = ",")

#Fixing the problem with some id_patient 
sum(is.na(CAM$id_patient))

CAM[CAM$episode_number=='N313987409',"id_patient"] <- 119935
CAM[CAM$episode_number=='N313672675',"id_patient"] <- 3141
CAM[CAM$episode_number=='N324525021'|CAM$episode_number=='N324908359'|CAM$episode_number=='N325015767',
    "id_patient"] <- 3144
CAM[CAM$episode_number=='N325137890',"id_patient"] <- 4537
CAM[CAM$episode_number=='N324242685',"id_patient"] <- 4539
CAM[CAM$episode_number=='N325080795',"id_patient"] <- 29099
CAM[CAM$episode_number=='N324943991',"id_patient"] <- 29102
CAM[CAM$episode_number=='N324780886'|CAM$episode_number=='N324860211'|CAM$episode_number=='N324877994'|CAM$episode_number=='N325485856',
    "id_patient"] <- 29112
CAM[CAM$episode_number=='N314140404',"id_patient"] <- 29115
CAM[CAM$episode_number=='N314094894',"id_patient"] <- 29119
CAM[CAM$episode_number=='N312822188'|CAM$episode_number=='N314142464'|CAM$episode_number=='N324312755'|CAM$episode_number=='N324334153'|CAM$episode_number=='N324340193'|CAM$episode_number=='N324799252'|CAM$episode_number=='N324752048',
    "id_patient"] <- 29123
CAM[CAM$episode_number=='N324307514'|CAM$episode_number=='N324323175'|CAM$episode_number=='N324343947',"id_patient"] <- 79083
CAM[CAM$episode_number=='N324877229',"id_patient"] <- 79085
CAM[CAM$episode_number=='N325229472'|CAM$episode_number=='N325310709'|CAM$episode_number=='N325479470',"id_patient"] <- 79090
CAM[CAM$episode_number=='N324843205',"id_patient"] <- 73210
CAM[CAM$episode_number=='N325062946',"id_patient"] <- 73215
CAM[CAM$episode_number=='N312830655'|CAM$episode_number=='N313067532'|CAM$episode_number=='N324230348',
    "id_patient"] <- 51246
CAM[CAM$episode_number=='N313524529'|CAM$episode_number=='N325622969',"id_patient"] <- 51248
CAM[CAM$episode_number=='N312748606',"id_patient"] <- 51249
CAM[CAM$episode_number=='N313301089'|CAM$episode_number=='N325278644',"id_patient"] <- 51253

sum(is.na(arc$id_patient))

arc[arc$Client.Date.of.birth==ymd('1932-08-13')&arc$Client.Date.of.death==ymd('2015-02-28'),"id_patient"] <- 3139
arc[arc$Client.Date.of.birth==ymd('1940-07-06')&arc$Client.Date.of.death==ymd('2024-02-25'),"id_patient"] <- 73215
#arc[arc$Client.Date.of.birth==ymd('1938-10-27')&arc$Client.Date.of.death==ymd('2018-03-29'),"id_patient"] <- 73216
arc[is.na(arc$id_patient),"id_patient"] <- 73216

sum(is.na(cont$id_patient))

cont[1983625:1983628,"id_patient"] <- 119931
cont[1983619:1983624,"id_patient"] <- 119930
cont[1481385:1481353,"id_patient"] <- 79091
cont[1481352:1481288,"id_patient"] <- 79090
cont[1481287:1481280,"id_patient"] <- 79087
cont[1481279:1481249,"id_patient"] <- 79085
cont[1481248,"id_patient"] <- 79083
cont[1415710:1415648,"id_patient"] <- 73216
cont[1415647:1415630,"id_patient"] <- 73215
cont[1415629:1415607,"id_patient"] <- 73212
cont[1021455:1021454,"id_patient"] <- 51255
cont[1021453:1021443,"id_patient"] <- 51253
cont[1021442,"id_patient"] <- 51249
cont[1021441:1021382,"id_patient"] <- 51248
cont[1021381:1021362,"id_patient"] <- 51246
cont[605367,"id_patient"] <- 29124
cont[605366:605302,"id_patient"] <- 29123
cont[605259:605301,"id_patient"] <- 29122
cont[605258:605256,"id_patient"] <- 29121
cont[605255:605235,"id_patient"] <- 29119
cont[605070:605234,"id_patient"] <- 29118
cont[605069,"id_patient"] <- 29117
cont[605068,"id_patient"] <- 29116
cont[605067:605040,"id_patient"] <- 29115
cont[604955:605039,"id_patient"] <- 29114
cont[604954:604898,"id_patient"] <- 29112
cont[604889:604897,"id_patient"] <- 29108
cont[604888:604877,"id_patient"] <- 29107
cont[604872:604876,"id_patient"] <- 29102
cont[604871:604854,"id_patient"] <- 29101
cont[604849:604853,"id_patient"] <- 29100
cont[604848:604810,"id_patient"] <- 29099
cont[112785:112546,"id_patient"] <- 4541
cont[112514:112545,"id_patient"] <- 4539
cont[80751:80719,"id_patient"] <- 3146
cont[80718,"id_patient"] <- 3145
cont[80700:80717,"id_patient"] <- 3144
cont[80699:80655,"id_patient"] <- 3143
cont[80649:80654,"id_patient"] <- 3141
cont[80648:80573,"id_patient"] <- 3139

sum(is.na(EDA$id_patient))

EDA[291566:291565,"id_patient"] <- 119943
EDA[291564,"id_patient"] <- 119942
EDA[291563:291562,"id_patient"] <- 119937
EDA[291561,"id_patient"] <- 119936
EDA[291560:291558,"id_patient"] <- 119931
EDA[291554:291557,"id_patient"] <- 119930
EDA[291553,"id_patient"] <- 119929
EDA[291552,"id_patient"] <- 119927
EDA[291551,"id_patient"] <- 119926
EDA[211433:211431,"id_patient"] <- 79091
EDA[211430:211424,"id_patient"] <- 79090
EDA[211423:211422,"id_patient"] <- 79087
EDA[211421:211420,"id_patient"] <- 79085
EDA[211419,"id_patient"] <- 79084
EDA[211418:211412,"id_patient"] <- 79083
EDA[211411:211409,"id_patient"] <- 79082
EDA[211265,"id_patient"] <- 79011
EDA[200981:200966,"id_patient"] <- 73216
EDA[200963,"id_patient"] <- 73212
EDA[200965,"id_patient"] <- 73215
EDA[200964,"id_patient"] <- 73213
EDA[200962:200961,"id_patient"] <- 73210
EDA[145737,"id_patient"] <- 51255
EDA[145736,"id_patient"] <- 51254
EDA[145735,"id_patient"] <- 51249
EDA[145734:145731,"id_patient"] <- 51248
EDA[86402:86401,"id_patient"] <- 29125
EDA[86400,"id_patient"] <- 29124
EDA[86399:86379,"id_patient"] <- 29123
EDA[86364:86378,"id_patient"] <- 29122
EDA[86363:86361,"id_patient"] <- 29121
EDA[86353:86360,"id_patient"] <- 29119
EDA[86352:86347,"id_patient"] <- 29118
EDA[86345:86346,"id_patient"] <- 29117
EDA[86344:86335,"id_patient"] <- 29116
EDA[86327:86334,"id_patient"] <- 29115
EDA[86326,"id_patient"] <- 29114
EDA[86325:86319,"id_patient"] <- 29112
EDA[86314:86318,"id_patient"] <- 29109
EDA[86313,"id_patient"] <- 29108
EDA[86312:86310,"id_patient"] <- 29107
EDA[86309:86308,"id_patient"] <- 29105
EDA[86301:86307,"id_patient"] <- 29103
EDA[86300:86299,"id_patient"] <- 29102
EDA[86297:86298,"id_patient"] <- 29101
EDA[86296:86293,"id_patient"] <- 29100
EDA[86292,"id_patient"] <- 29099
EDA[14243:14240,"id_patient"] <- 4541
EDA[14237:14239,"id_patient"] <- 4539
EDA[14236,"id_patient"] <- 4537
EDA[10158:10150,"id_patient"] <- 3146
EDA[10149,"id_patient"] <- 3145
EDA[10148:10143,"id_patient"] <- 3144
EDA[10142:10133,"id_patient"] <- 3141
EDA[10132:10099,"id_patient"] <- 3139
EDA[10097:10098,"id_patient"] <- 3138
EDA[10096,"id_patient"] <- 3137

sum(is.na(inter$id_patient))

inter[25060:25057,"id_patient"] <- 79090
inter[24482:24481,"id_patient"] <- 76923
inter[23662:23661,"id_patient"] <- 73216
inter[23653:23660,"id_patient"] <- 73215
inter[16318:16317,"id_patient"] <- 51248
inter[9448:9445,"id_patient"] <- 29123
inter[9444:9443,"id_patient"] <- 29118
inter[9442:9441,"id_patient"] <- 29101
inter[9439:9440,"id_patient"] <- 29100
inter[1104:1103,"id_patient"] <- 3143
inter = inter[-c(10843:10844),] # In interRAI but not in master table, then delete

sum(is.na(IPA$id_patient))

IPA[2107230,"id_patient"] <- 119943
IPA[2107229:2107225,"id_patient"] <- 119942
IPA[2107221:2107224,"id_patient"] <- 119937
IPA[2107220:2107217,"id_patient"] <- 119935
IPA[2107207:2107216,"id_patient"] <- 119931
IPA[2107187:2107206,"id_patient"] <- 119930
IPA[2107186:2107182,"id_patient"] <- 119929
IPA[2107181,"id_patient"] <- 119927
IPA[2107180:2107178,"id_patient"] <- 119926
IPA[1545170:1545148,"id_patient"] <- 79091
IPA[1545062:1545147,"id_patient"] <- 79090
IPA[1545061:1545053,"id_patient"] <- 79087
IPA[1545008:1545052,"id_patient"] <- 79085
IPA[1545007:1545004,"id_patient"] <- 79084
IPA[1544951:1545003,"id_patient"] <- 79083
IPA[1544950:1544941,"id_patient"] <- 79082
IPA[1543963,"id_patient"] <- 79011
IPA[1471768:1471671,"id_patient"] <- 73216
IPA[1471656:1471670,"id_patient"] <- 73215
IPA[1471655,"id_patient"] <- 73213
IPA[1471654:1471635,"id_patient"] <- 73212
IPA[1471634:1471633,"id_patient"] <- 73210
IPA[1060693:1060684,"id_patient"] <- 51255
IPA[1060667:1060683,"id_patient"] <- 51254
IPA[1060666:1060654,"id_patient"] <- 51253
IPA[1060650:1060653,"id_patient"] <- 51249
IPA[1060649:1060635,"id_patient"] <- 51248
IPA[1060620:1060634,"id_patient"] <- 51246
IPA[622293:622281,"id_patient"] <- 29125
IPA[622279:622280,"id_patient"] <- 29124
IPA[622278:622089,"id_patient"] <- 29123
IPA[622042:622088,"id_patient"] <- 29122
IPA[622041:622026,"id_patient"] <- 29121
IPA[621977:622025,"id_patient"] <- 29119
IPA[621976:621889,"id_patient"] <- 29118
IPA[621883:621888,"id_patient"] <- 29117
IPA[621882:621869,"id_patient"] <- 29116
IPA[621836:621868,"id_patient"] <- 29115
IPA[621835:621834,"id_patient"] <- 29114
IPA[621823:621833,"id_patient"] <- 29113
IPA[621822:621763,"id_patient"] <- 29112
IPA[621757:621762,"id_patient"] <- 29111
IPA[621756:621753,"id_patient"] <- 29109
IPA[621747:621752,"id_patient"] <- 29108
IPA[621746:621736,"id_patient"] <- 29107
IPA[621735,"id_patient"] <- 29105
IPA[621711:621734,"id_patient"] <- 29103
IPA[621710:621698,"id_patient"] <- 29102
IPA[621683:621697,"id_patient"] <- 29101
IPA[621682:621667,"id_patient"] <- 29100
IPA[621639:621666,"id_patient"] <- 29099
IPA[105249:105191,"id_patient"] <- 4541
IPA[105165:105190,"id_patient"] <- 4539
IPA[105164:105160,"id_patient"] <- 4537
IPA[105159,"id_patient"] <- 4535
IPA[74326:74325,"id_patient"] <- 3147
IPA[74252:74324,"id_patient"] <- 3146
IPA[74251:74245,"id_patient"] <- 3145
IPA[74197:74244,"id_patient"] <- 3144
IPA[74196:74195,"id_patient"] <- 3143
IPA[74146:74194,"id_patient"] <- 3141
IPA[74145:74044,"id_patient"] <- 3139
IPA[74039:74043,"id_patient"] <- 3138
IPA[74038:74028,"id_patient"] <- 3137
IPA[74021:74027,"id_patient"] <- 3136

sum(is.na(OPA$id_patient))
sum(is.na(lab$id_patient))
sum(is.na(Med$id_patient)) # There are 55 id_patients in Pharmacy data but not in the master. 
Med <- na.omit(Med[Med$id_patient != "NA", ])
Med$generic_name = tolower(Med$generic_name)

#Name of the blood tests: to clean them---------------------------------------------
blood_tests = lab %>% group_by(result_test_name) %>% summarise('freq'=n())
write_xlsx(blood_tests, "blood_tests_names.xlsx")
#-----------------------------------------------------------------------------------

#blood_tests = read_xlsx("blood_tests_names_bef.xlsx", sheet = "Sheet1", trim_ws = FALSE)
#blood_tests = blood_tests[,-2]

lab = merge(lab, blood_tests, by = "result_test_name", all.x = TRUE)
lab = lab %>% filter(!is.na(test_name))

#blood1 = lab %>% group_by(test_name) %>% summarise('tot'=n())
#write_xlsx(blood1, "blood_tests_names.xlsx")

#2. Identify if the patients has a diagnosis of dementia or not. 
# To do this, I will use the datasets 'inter', 'Med' and IP. 
inter$DateSigned = ymd_hms(inter$DateSigned)
Dem1 = inter %>% group_by(id_patient) %>% summarise("sum"=sum(Response, na.rm = TRUE)) 
Dem1$Dementia = ifelse(Dem1$sum>=1, 'Dementia', 'No dementia')
Dem1_1 = inter %>% filter(Response>=1) %>% group_by(id_patient)%>% summarise('date_diagnosis' = min(DateSigned))

Dem1 = merge(Dem1, Dem1_1, by = 'id_patient', all.x = T)

Med$dispense_date = ymd_hms(Med$dispense_date)

Med$Dem_score = 0
Med$Dem_score <- ifelse(grepl('donepezil', Med$generic_name, ignore.case = TRUE) | 
                            grepl('rivastigmine', Med$generic_name, ignore.case = TRUE)|
                          grepl('galantamine', Med$generic_name, ignore.case = TRUE)|
                          grepl('memantine', Med$generic_name, ignore.case = TRUE), 
                          1, Med$Dem_score)
#ifelse(Med$generic_name=='Donepezil hydrochloride'|Med$generic_name=='Rivastigmine',1,0)
Dem2 = Med %>% group_by(id_patient) %>% summarise("sum"=sum(Dem_score, na.rm = TRUE))
Dem2$Dementia = ifelse(Dem2$sum>=1, 'Dementia', 'No dementia')
Dem2_1 = Med %>% filter(Dem_score==1) %>% group_by(id_patient)%>% summarise('date_diagnosis' = min(dispense_date, na.rm = TRUE))
#Inf because all values are NA
Dem2_1[Dem2_1$date_diagnosis==Inf,'date_diagnosis'] <- NA

Dem2 = merge(Dem2, Dem2_1, by = 'id_patient', all.x = T)

#Only to get the codes for dementia---------------------------------------------------------
IPA$clinical_code_description = tolower(IPA$clinical_code_description)
IPA$dem = 0 
IPA$dem <- ifelse(grepl('dementia', IPA$clinical_code_description, ignore.case = TRUE)| 
                    grepl('alzheimer', IPA$clinical_code_description, ignore.case = TRUE), 
                        1, IPA$dem)
f1 = subset(IPA, dem == 1)
write.csv(f1, 'dem_codes.csv')
#-------------------------------------------------------------------------------------------

IPA$admission_datetime = ymd_hms(IPA$admission_datetime)
IPA$discharge_datetime = ymd_hms(IPA$discharge_datetime)
IPA$coded_date = ymd_hms(IPA$coded_date)

IPA$Dem_score = ifelse(IPA$clinical_code=='F00'|IPA$clinical_code=='F000'|IPA$clinical_code=='F0000'|
                         IPA$clinical_code=='F001'|IPA$clinical_code=='F002'|IPA$clinical_code=='F0020'|
                         IPA$clinical_code=='F0021'|IPA$clinical_code=='F009'|IPA$clinical_code=='F0090'|
                         IPA$clinical_code=='F0091'|IPA$clinical_code=='F01'|IPA$clinical_code=='F010'|
                         IPA$clinical_code=='F011'|IPA$clinical_code=='F0110'|IPA$clinical_code=='F012'|
                         IPA$clinical_code=='F0120'|IPA$clinical_code=='F013'|IPA$clinical_code=='F0131'|
                         IPA$clinical_code=='F018'|IPA$clinical_code=='F0180'|IPA$clinical_code=='F019'|
                         IPA$clinical_code=='F0190'|IPA$clinical_code=='F0191'|IPA$clinical_code=='F02'|
                         IPA$clinical_code=='F020'|IPA$clinical_code=='F0200'|IPA$clinical_code=='F0201'|
                         IPA$clinical_code=='F021'|IPA$clinical_code=='F022'|IPA$clinical_code=='F023'|
                         IPA$clinical_code=='F0230'|IPA$clinical_code=='F0231'|IPA$clinical_code=='F024'|
                         IPA$clinical_code=='F028'|IPA$clinical_code=='F0280'|IPA$clinical_code=='F0281'|
                         IPA$clinical_code=='F03'|IPA$clinical_code=='F0300'|IPA$clinical_code=='F0301'|
                         IPA$clinical_code=='F051'|IPA$clinical_code=='G300'|IPA$clinical_code=='G301'|
                         IPA$clinical_code=='G308'|IPA$clinical_code=='G309'|IPA$clinical_code=='U791',1,0)

Dem3 = IPA %>% group_by(id_patient) %>% summarise("sum"=sum(Dem_score, na.rm = TRUE))
Dem3$Dementia = ifelse(Dem3$sum>=1, 'Dementia', 'No dementia')
IPA$Dementia_date = ifelse(!is.na(IPA$coded_date), IPA$coded_date, IPA$admission_datetime)
IPA$Dementia_date = as.POSIXct(IPA$Dementia_date, tz=Sys.timezone())
Dem3_1 = IPA %>% filter(Dem_score==1) %>% group_by(id_patient)%>% 
  summarise('date_diagnosis' = min(Dementia_date))

Dem3 = merge(Dem3, Dem3_1, by = 'id_patient', all.x = T)

venn.diagram(list(Dem1$id_patient, Dem2$id_patient, Dem3$id_patient), 'venn.png', imagetype = 'png',
             category.names = c('interRAI\nn=14233', 'Pharmacy\nn=112746', "Inpatient\nn=84240"), output=FALSE, 
             cat.cex = 0.6, lwd = 1, cex = 1, rotation = 1, scaled = TRUE)

venn.diagram(list(Dem1[Dem1$Dementia=='Dementia',]$id_patient, Dem2[Dem2$Dementia=='Dementia',]$id_patient,
                  Dem3[Dem3$Dementia=='Dementia',]$id_patient), 'venn_dem.png', imagetype = 'png',
             category.names = c('interRAI\nn=14233', 'Pharmacy\nn=112746', "Inpatient\nn=84240"), output=FALSE, 
             cat.cex = 0.6, lwd = 1, cex = 1, rotation = 1, scaled = TRUE)

#display_venn(list(Dem1$id_patient, Dem2$id_patient, Dem3$id_patient))

Dem = list(Dem1, Dem2, Dem3)
Dem = Dem %>% reduce(full_join, by='id_patient')
Dem$Diagnosis_score = rowSums(Dem[,c(2,5,8)], na.rm = T)
Dem$Diagnosis = ifelse(Dem$Diagnosis_score >= 1, 'Dementia', 'No dementia')
Dem$Date_Diagnosis = apply(Dem[,c(4,7,10)],1,min,na.rm=T)

Dem[Dem$Diagnosis =='Dementia'&is.na(Dem$Date_Diagnosis),'Date_Diagnosis'] <- '2014-06-01' # get those with dementia but not dates

Dem = Dem[,c(1,12,13)]

Dem$Date_Diagnosis = ymd_hms(Dem$Date_Diagnosis, truncated = 3)

#3. Prepare 'soc' data set for case-control extraction by age (+-2 years).  

soc = soc[-137522,-7]

#soc$date_of_death = ifelse(soc$date_of_death == 'NULL', NA, soc$date_of_death)
#soc$date_of_death = as.POSIXct(soc$date_of_death, tz=Sys.timezone())


soc$Age_to_now = trunc((ym(soc$date_of_birth) %--% ymd('2024-03-19'))/years(1))
#soc$Age_group = cut(soc$Age_to_now, c(0,64,74,84,max(soc$Age_to_now)), 
#                    labels = c('<64', '65-74', '75-84', '>=85'))

#Using StatsNZ classification

soc$Ethnicity = ifelse(soc$prioritised_ethnicity_description == 'Chinese'|soc$prioritised_ethnicity_description == 'Indian'|soc$prioritised_ethnicity_description == 'Asian not further defined'|
                         soc$prioritised_ethnicity_description == 'Other Asian'| soc$prioritised_ethnicity_description == 'Southeast Asian', 'Asian',
                       ifelse(soc$prioritised_ethnicity_description == 'NZ Maori', 'Māori',
                              ifelse(soc$prioritised_ethnicity_description == 'European not further defined'|soc$prioritised_ethnicity_description == 'NZ European'|
                                       soc$prioritised_ethnicity_description == 'Other European', 'European', 
                                     ifelse(soc$prioritised_ethnicity_description == 'Middle Eastern'| soc$prioritised_ethnicity_description == 'Latin American / Hispanic'|
                                              soc$prioritised_ethnicity_description == 'African', 'MiddleEastern_LatinAmerican_African',
                                            ifelse(soc$prioritised_ethnicity_description == "Don't know"|soc$prioritised_ethnicity_description == "Refused to answer"|soc$prioritised_ethnicity_description == "Response unidentifiable"|soc$prioritised_ethnicity_description == "Not stated",'Residual Categories',
                                                   ifelse(soc$prioritised_ethnicity_description == "Other ethnicity"|soc$prioritised_ethnicity_description == "Other (retired on 1/07/2009)", "Other", 'Pacific'))))))


#soc$Ethnicity = ifelse(soc$prioritised_ethnicity_description == 'Chinese'|soc$prioritised_ethnicity_description == 'Indian'|
#                         soc$prioritised_ethnicity_description == 'Other Asian'| soc$prioritised_ethnicity_description == 'Southeast Asian', 'Asian',
#                       ifelse(soc$prioritised_ethnicity_description == 'NZ Maori', 'Māori',
#                              ifelse(soc$prioritised_ethnicity_description == 'European not further defined'|soc$prioritised_ethnicity_description == 'NZ European'|
#                                       soc$prioritised_ethnicity_description == 'Other European', 'NZ European', 
#                                     ifelse(soc$prioritised_ethnicity_description == 'Middle Eastern'| soc$prioritised_ethnicity_description == 'Not stated'|
#                                              soc$prioritised_ethnicity_description == 'Other ethnicity', 'Other', 'Pacific'))))

soc = merge(soc, Dem, by = 'id_patient', all.x = TRUE)

missing_all_dat = setdiff(soc$id_patient, c(IPA$id_patient, EDA$id_patient,
                                            OPA$id_patient, cont$id_patient,
                                            CAM$id_patient, lab$id_patient,
                                            Med$id_patient, arc$id_patient))

soc <- soc %>%
  filter(!id_patient %in% missing_all_dat)

# Playing with dates
CAM$assessment_datetime = dmy_hm(CAM$assessment_datetime)

f1 = IPA %>% group_by(id_patient) %>% summarise('min_date_IPA'=min(admission_datetime, na.rm = TRUE))
f2 = EDA %>% group_by(id_patient) %>% summarise('min_date_EDA'=min(arrived_datetime, na.rm = TRUE))
f3 = OPA %>% group_by(id_patient) %>% summarise('min_date_OPA'=min(appointment_start_date, na.rm = TRUE))
f4 = cont %>% group_by(id_patient) %>% summarise('min_date_cont'=min(contact_start_datetime, na.rm = TRUE))
f5 = CAM %>% group_by(id_patient) %>% summarise('min_date_CAM'=min(assessment_datetime, na.rm = TRUE))
f6 = lab %>% group_by(id_patient) %>% summarise('min_date_lab'=min(request_date, na.rm = TRUE))
f7 = Med %>% group_by(id_patient) %>% summarise('min_date_Med'=min(dispense_date, na.rm = TRUE))
f7[f7$min_date_Med==Inf,'min_date_Med'] <- NA # ymd('2014-06-01')
f8 = arc %>% group_by(id_patient) %>% summarise('min_date_arc'=min(`Service.Start`, na.rm = TRUE))

ft = list(f1, f2, f3, f4, f5, f6, f7, f8)
ft = ft %>% reduce(full_join, by='id_patient')

ft$min_date = apply(ft[,-1],1,FUN=min, na.rm = TRUE)
ft$min_date = ifelse(is.na(ft$min_date), '2014-06-01', ft$min_date)
ft$years_to_now = as.numeric(difftime(ymd('2024-03-19'), ft$min_date, units = 'days'))/365.25 #Years to now
  #interval(ft$min_date, ymd('2024-03-19')) / months(1)
ft = ft[,c(1,11)]

soc = merge(soc, ft, by = 'id_patient', all.x = TRUE)
soc[!is.na(soc$Diagnosis)&is.na(soc$years_to_now),]
#soc = soc[-73013,] #This patient only have information in interRAI but not in the other datasets. I removed them.
soc_all = soc

soc_pre = soc %>% filter(!is.na(years_to_now))
soc_pre[duplicated(soc_pre$id_patient),]
soc_pre = soc_pre[-101533,]

soc_pre$Diagnosis = ifelse(is.na(soc_pre$Diagnosis), 'No dementia', soc_pre$Diagnosis)

# Selection of cases and controls
soc_pre %>%
  split(.$Diagnosis) %>%
  list2env(envir = .GlobalEnv)

`No dementia`$FILTER <- FALSE
`No dementia`$case_index <- NA

for (i in seq_len(nrow(Dementia))) {
  set.seed(666)
  #set.seed(777)
  x <- which(between(`No dementia`$Age_to_now, Dementia$Age_to_now[i] - 2, Dementia$Age_to_now[i] + 2) &
               between(`No dementia`$years_to_now, Dementia$years_to_now[i] - 2, Dementia$years_to_now[i] + 2) &
               is.na(`No dementia`$case_index)& 
               !`No dementia`$FILTER)
  #print(x)
  #print(length(x))
  if (length(x)>= 1){
    selected_control <- sample(x, min(1, length(x))) # Change depending how many controls per case. 
    #print(selected_control)
    `No dementia`$FILTER[selected_control] <- TRUE
    `No dementia`$case_index[selected_control] <- Dementia$id_patient[i]
    `No dementia`$Date_Diagnosis[selected_control] <- Dementia$Date_Diagnosis[i]
  }
}

soc = bind_rows(Dementia, `No dementia`) %>% filter(FILTER | is.na(FILTER)) %>% 
  select(-c(FILTER,case_index, years_to_now))

Dem_NoDem = soc[,c(1,10:11)]

#4. Keep only the information before the diagnosis of dementia.

keep_before = function(original_set, dementia_date, var_date_name, days_bef = 0){
  dat1 = merge(original_set, dementia_date, by = 'id_patient', all.y = T)
  dat1[,var_date_name] = ymd_hms(dat1[,var_date_name], truncated = 3)
  dat1$Date_Diagnosis = ymd_hms(dat1$Date_Diagnosis, truncated = 3)
  dat1$Diff = difftime(dat1$Date_Diagnosis, dat1[,var_date_name], units = 'days')
  dat2 = dat1 %>% filter(Diff>days_bef)#|is.na(Diff))
  return(dat2)
}

IPA_bef = keep_before(IPA, Dem_NoDem, "admission_datetime", days_bef = 1095)
EDA_bef = keep_before(EDA, Dem_NoDem, "arrived_datetime", days_bef = 1095)
OPA_bef = keep_before(OPA, Dem_NoDem, "appointment_start_date", days_bef = 1095)
cont_bef = keep_before(cont, Dem_NoDem, "contact_start_datetime", days_bef = 1095)
CAM_bef = keep_before(CAM, Dem_NoDem, "assessment_datetime", days_bef = 1095)
inter_bef = keep_before(inter, Dem_NoDem, "DateSigned")
lab_bef = keep_before(lab, Dem_NoDem, "test_result_datetime", days_bef = 1095)
Med_bef = keep_before(Med, Dem_NoDem, "dispense_date", days_bef = 1095)
arc_bef = keep_before(arc, Dem_NoDem, "Service.Start", days_bef = 1095)

# Finding the id with no information in the sequences:
id_patient_seq = soc$id_patient

id_patient_seq = Reduce(union,list(IPA_bef$id_patient, EDA_bef$id_patient,
                         OPA_bef$id_patient, cont_bef$id_patient,
                         CAM_bef$id_patient, lab_bef$id_patient,
                         Med_bef$id_patient, arc_bef$id_patient))

#ONLY FOR the test IMPORTANT++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#test = read.csv('array.csv', header = TRUE, sep = ',')
#soc_test = merge(test, soc, by = 'id_patient', all.x = TRUE)

#id_patient_seq = soc_test$id_patient

missing_all_dat = setdiff(id_patient_seq, c(IPA_bef$id_patient, EDA_bef$id_patient,
                                        OPA_bef$id_patient, cont_bef$id_patient,
                                        CAM_bef$id_patient, lab_bef$id_patient,
                                        Med_bef$id_patient, arc_bef$id_patient))

#soc_test1 <- soc_test %>%
#  filter(!id_patient %in% missing_all_dat)

soc_test1 <- soc %>%
  filter(!id_patient %in% missing_all_dat)

id_patient_seq = soc_test1$id_patient
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# I decided to include all, include those with no sequences. They could have information in the other
# dataset. Still, this is something I have to asks. If I remove them, I will lose some cases and controls 
# and the design will be affected. SOLVED

#-------------------------------------------------------------------------------------------------------------
IPA_bef_id = IPA_bef %>% group_by(id_patient) %>% summarise('n'=n())
CAM_bef_id = CAM_bef %>% group_by(id_patient) %>% summarise('n'=n())
lab_bef_id = lab_bef %>% group_by(id_patient) %>% summarise('n'=n())

#id with no sequences to remove from the study
missing_seq = setdiff(id_patient, c(IPA_bef_id$id_patient, CAM_bef_id$id_patient, lab_bef_id$id_patient))

#id with sequences
id_patient_seq = intersect(c(IPA_bef_id$id_patient, CAM_bef_id$id_patient, lab_bef_id$id_patient), id_patient)

#Removing those with no sequences 
remove_ids <- function(dataset) {
  dt = dataset[!dataset$id_patient %in% missing_seq,]
  num = sum(dataset$id_patient %in% missing_seq)
  print(num)
  return(dt)
}

soc <- remove_ids(soc)
IPA_bef = remove_ids(IPA_bef)
EDA_bef = remove_ids(EDA_bef)
OPA_bef = remove_ids(OPA_bef)
cont_bef = remove_ids(cont_bef)
CAM_bef = remove_ids(CAM_bef)
lab_bef = remove_ids(lab_bef)
Med_bef = remove_ids(Med_bef)
arc_bef = remove_ids(arc_bef)
#--------------------------------------------------------------------------------------------------------------

#5. Extract the static features 

#4a. For IP admissions, we need to get hospitalisation patients, defined as
# their length of stay per admission is more than 24 hours. 

IPA_bef$Dif_ad_di = as.numeric(difftime(IPA_bef$discharge_datetime, IPA_bef$admission_datetime, units = 'days'))
IPA_bef1 = IPA_bef %>% filter(Dif_ad_di>=1 & coding_sequence_number == 1)
stat1 = IPA_bef1 %>% group_by(id_patient) %>% summarise('number_admissions'= n())

soc1 = merge(soc, stat1, by = 'id_patient', all.x = TRUE)
#soc1 = merge(soc_test1, stat1, by = 'id_patient', all.x = TRUE) #Only test
#soc1$tot_los = ifelse(is.na(soc1$tot_los),0,soc1$tot_los)
soc1$number_admissions = ifelse(is.na(soc1$number_admissions),0,soc1$number_admissions)

#Sequence of length of stay
nac = soc[,1:2]
nac$date_of_birth = ym(nac$date_of_birth)

#FOR TEST+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nac = soc_test1[,1:2]
nac$date_of_birth = ym(nac$date_of_birth)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

IPA_bef1 = merge(IPA_bef1, nac, by = 'id_patient', all.x = TRUE)
IPA_bef1$Timestamp = as.numeric(difftime(IPA_bef1$admission_datetime, 
                                         IPA_bef1$date_of_birth, units = 'days'))
los = IPA_bef1[,c(1,19,17)]
los = los %>% filter(!is.na(Timestamp))
los$Dif_ad_di = scale(los$Dif_ad_di)
los$Timestamp = scale(los$Timestamp)

los %>% ggplot(aes(Timestamp, Dif_ad_di, group = id_patient,
                        color = id_patient)) + geom_line()

missing = setdiff(id_patient_seq, los$id_patient)
patient = data.frame(id_patient=missing)
los = bind_rows(los, patient)
los[is.na(los)]<-0

#Sequence of ICD-10 codes for each visit (Longitudinal)
#icd10 = read_xlsx("ICD10_Vocabulary.xlsx") #To change with the new data

#icd10_train <- IPA_bef %>%
#  distinct(clinical_code) %>%
#  mutate(Number_ICD10 = row_number())

IPA_bef = merge(IPA_bef, icd10_train, by = 'clinical_code', all.x = TRUE)

# Do not run this ------------------------------------------------------------------------
sequences <- IPA_bef %>% filter(Dif_ad_di>=1) %>%
  group_by(id_patient, admission_datetime) %>%
  summarize(ICD_Sequence = list(clinical_code)) %>%
  mutate(visit_number = paste("visit",row_number(), sep = "_"))

sequences1 <- pivot_wider(sequences,id_cols = "id_patient", names_from = "visit_number", 
                          values_from = "ICD_Sequence")

####Another format just in case is needed
sequences3 <- IPA_bef %>% filter(Dif_ad_di>=1) %>%
  group_by(id_patient, admission_datetime) %>%
  summarize(ICD_Sequence = list(Number_ICD10)) 

for (i in 1:nrow(sequences3)) {
  sequences3$ICD_Sequence[i] <- paste(unlist(sequences3$ICD_Sequence[i]), collapse = " ")
}
sequences3$ICD_Sequence = as.character(sequences3$ICD_Sequence)

sequences3 = merge(sequences3, nac, by = 'id_patient', all.x = TRUE)
sequences3$Timestamp = as.numeric(difftime(sequences3$admission_datetime, 
                                           sequences3$date_of_birth, units = 'days'))
sequences3 = sequences3[,c(1,5,3)]
#------------------------------------------------------------------------------------------

#Another way 
#IPA_bef$admission_datetime = ymd_hms(IPA_bef$admission_datetime)
#IPA_bef$coded_date = ymd_hms(IPA_bef$coded_date)

IPA_bef$coded_date = ifelse(!is.na(IPA_bef$coded_date),IPA_bef$coded_date,IPA_bef$admission_datetime)
IPA_bef$coded_date = as.POSIXct(IPA_bef$coded_date, tz=Sys.timezone())
IPA_bef = merge(IPA_bef, nac, by = 'id_patient', all.x = TRUE)
IPA_bef$Timestamp = as.numeric(difftime(IPA_bef$coded_date, 
                                        IPA_bef$date_of_birth, units = 'days'))
Visits = IPA_bef %>% filter(Dif_ad_di>=1)
Visits = Visits[,c(1,20,18)]
Visits = Visits %>% filter(!is.na(Timestamp))
Visits = Visits %>% filter(!is.na(Number_ICD10))
Visits$Timestamp = scale(Visits$Timestamp)

missing = setdiff(id_patient_seq, Visits$id_patient)
patient = data.frame(id_patient=missing)
Visits = bind_rows(Visits, patient)
Visits[is.na(Visits)]<-0 #Ask about the missing data


#4b. For ED, extract number of time in ED before dementia. 
EDA_bef1 = EDA_bef %>% group_by(id_patient) %>% summarise('number_ED_attendance'=n())

soc2 = merge(soc1, EDA_bef1, by = 'id_patient', all.x = TRUE)
soc2$number_ED_attendance = ifelse(is.na(soc2$number_ED_attendance),0,soc2$number_ED_attendance)

#4c. For OP, extract the number of appointments by specialty
OPA_bef1 = OPA_bef %>% group_by(id_patient, outpatient_specialty_desc) %>%
  summarise("Number_appoint"=n())

OPA_bef$attendance_status = ifelse(OPA_bef$attendance_status_desc=='Did not wait'|
                                     OPA_bef$attendance_status_desc=='Not Specified'|
                                     OPA_bef$attendance_status_desc=='Service Not Delivered/Incomplete','Other',
                                   OPA_bef$attendance_status_desc)

OPA_bef3 = OPA_bef %>% group_by(id_patient, attendance_status) %>%
  summarise("Number_attendance"=n())

OPA_bef2 <- pivot_wider(OPA_bef1,id_cols = "id_patient", names_from = "outpatient_specialty_desc", 
                          values_from = "Number_appoint")

OPA_bef4 <- pivot_wider(OPA_bef3,id_cols = "id_patient", names_from = "attendance_status", 
                        values_from = "Number_attendance")

OPA_bef5 = merge(OPA_bef2, OPA_bef4, by='id_patient', all = T)
colnames(OPA_bef5) <- c('id_patient',paste('OP',colnames(OPA_bef5)[-1],sep="_"))
OPA_bef5[is.na(OPA_bef5)]<-0

soc3 = merge(soc2, OPA_bef5, by = 'id_patient', all.x = TRUE)

#4d. For Contact, extract number of contact occurred, planned or cancelled. 
cont_bef1 = cont_bef %>% group_by(id_patient, contact_status_desc) %>% summarise("Contact_status"=n())
cont_bef1 <- pivot_wider(cont_bef1,id_cols = "id_patient", names_from = "contact_status_desc", 
                        values_from = "Contact_status")
colnames(cont_bef1) <- c('id_patient',paste('Contact',colnames(cont_bef1)[-1],sep="_"))
cont_bef1[is.na(cont_bef1)]<-0

soc4 = merge(soc3, cont_bef1, by = 'id_patient', all.x = TRUE)

#4e. For CAM, extract number of assessments, number of delirium screened, number non-consecutive delirium 

CAM_bef1 = CAM_bef %>% group_by(id_patient) %>% 
  summarise('number_CAM'=n(),'number_admissions_cam'=n_distinct(episode_number)) 

CAM_bef2 = CAM_bef %>% filter(CAM_Score_3._Evidence_Focus=='Yes') %>%
  group_by(id_patient) %>% summarise('Number_positive_delirium' = n())

CAM_bef2 = merge(CAM_bef1, CAM_bef2, by = 'id_patient', all.x = T)
CAM_bef2[is.na(CAM_bef2)]<-0

soc5 = merge(soc4, CAM_bef2, by = 'id_patient', all.x = TRUE)

# Do no run this 
#--------------------------------------------------------------------------------------------------
#Calculating the timestamp
CAM_bef = CAM_bef %>% group_by(id_patient) %>% mutate(CAM = paste("CAM", row_number(), sep = "_"))
sequence2 <- pivot_wider(CAM_bef,id_cols = "id_patient", names_from = "CAM", 
                        values_from = "total_score")
#--------------------------------------------------------------------------------------------------

###Sequence with timestamp
CAM_bef = merge(CAM_bef, nac, by = 'id_patient', all.x = TRUE)
CAM_bef$Timestamp = as.numeric(difftime(CAM_bef$assessment_datetime, 
                                        CAM_bef$date_of_birth, units = 'days'))

CAM_seq = CAM_bef %>% group_by(id_patient, episode_number) %>% 
  mutate('delirium_pos'=if_else(total_score>=3 & CAM_Score_3._Evidence_Focus == 'Yes', 1, 0))

CAM_seq1 = CAM_seq %>% group_by(id_patient, episode_number) %>% 
  summarise('num_del'=sum(delirium_pos))

CAM_seq2 = CAM_seq %>% group_by(id_patient, episode_number) %>% 
  summarise('Timestamp'=min(Timestamp))

CAM_seq3 = merge(CAM_seq1, CAM_seq2, by = c('id_patient', 'episode_number'))

CAM_seq3 = CAM_seq3[,c(1,4,3)]
CAM_seq3 = CAM_seq3 %>% filter(!is.na(Timestamp))
CAM_seq3$num_del = scale(CAM_seq3$num_del)
CAM_seq3$Timestamp = scale(CAM_seq3$Timestamp)

#sequences4 = CAM_bef[,c(1,16,10)]

missing = setdiff(id_patient_seq, CAM_seq3$id_patient)
patient = data.frame(id_patient=missing)
CAM_seq3 = bind_rows(CAM_seq3, patient)
CAM_seq3[is.na(CAM_seq3)]<-0 #Ask about the missing data

#4f. For blood tests, extract all sequences. First remove those rows when the test result is
# not numeric. 

lab_bef$test_result = as.numeric(gsub("[<>]","",lab_bef$test_result))
lab_bef = lab_bef %>% filter(!is.na(test_result))

# Make the test names consistent 

#lab_bef$result_test_name = ifelse(lab_bef$result_test_name == "WBC - White Cell Count","WBC",
#                                  ifelse(lab_bef$result_test_name == 'RBC - Red Cell Count',"RBC",
#                                         ifelse(lab_bef$result_test_name == 'Nucleated RBCs'|lab_bef$result_test_name == 'Nucleated Red Cell Count'|lab_bef$result_test_name == 'NRBCS'|lab_bef$result_test_name == 'Nucleated RBC','RBC (nucleated)',
#                                                ifelse(lab_bef$result_test_name == 'Platelet Count','Platelets',
#                                                       ifelse(lab_bef$result_test_name == 'Mean Cell Volume'|lab_bef$result_test_name == 'MCV - Mean Cell Volume','MCV',
#                                                              ifelse(lab_bef$result_test_name == 'Mean Cell Haemoglobin'|lab_bef$result_test_name == 'MCH - Mean Cell Haemoglobin','MCH',
#                                                                     ifelse(lab_bef$result_test_name == 'Hct - Haematocrit'|lab_bef$result_test_name == "HCT",'Haematocrit',
#                                                                            ifelse(lab_bef$result_test_name == "Hb  - Haemoglobin", 'Haemoglobin', 
#                                                                                   ifelse(lab_bef$result_test_name == 'B12', "Vitamin B12", 
#                                                                                          ifelse(lab_bef$result_test_name == 'Free T3','T3 (free)',
#                                                                                                 ifelse(lab_bef$result_test_name == 'Free T4', 'T4 (free)',
#                                                                                                        ifelse(lab_bef$result_test_name == 'Neutrophil/Lymphocyte Ratio', 'Neutrophil Lymphocyte Ratio', 
#                                                                                                               ifelse(lab_bef$result_test_name == ' Monocytes Percentage', 'Monocytes',
#                                                                                                                      ifelse(lab_bef$result_test_name == ' Neutrophils', 'Neutrophils',
#                                                                                                                             ifelse(lab_bef$result_test_name == 'Neutrophils (Band)', 'Neutrophils (band)',
#                                                                                                                                    ifelse(lab_bef$result_test_name == '24Hr Phosphate', 'Phosphate 24Hr',
#                                                                                                                                           ifelse(lab_bef$result_test_name == 'Adjusted Calcium', 'Calcium (albumin adjusted)',
#                                                                                                                                                  ifelse(lab_bef$result_test_name == 'Alk. Phosphatase', 'Alkaline phosphatase',
#                                                                                                                                                         ifelse(lab_bef$result_test_name == 'C-Reactive Protein', 'CRP',
#                                                                                                                                                                ifelse(lab_bef$result_test_name == 'Fluid Glucose'|lab_bef$result_test_name == 'Glucose other', 'Glucose',
#                                                                                                                                                                       ifelse(lab_bef$result_test_name == 'Serum Folate', 'Folate - serum', 
#                                                                                                                                                                              ifelse(lab_bef$result_test_name == 'Hct', 'Haematocrit', 
#                                                                                                                                                                                     ifelse(lab_bef$result_test_name == 'MPV - Mean Platelet Volume', 'MPV', 
#                                                                                                                                                                                            ifelse(lab_bef$result_test_name == 'Reticulocytes Haemoglobin', 'RET-He',
#                                                                                                                                                                                                   ifelse(lab_bef$result_test_name == 'Serum B12', 'Vitamin B12 - serum',
#                                                                                                                                                                                                          ifelse(lab_bef$result_test_name == 'Uric Acid', 'Urate',lab_bef$result_test_name))))))))))))))))))))))))))

#Removing duplicates
#lab_bef = lab_bef %>% 
#  filter(!duplicated(cbind(id_patient, test_result_datetime, result_test_name, test_result)))

lab_bef = lab_bef %>% distinct(id_patient, test_result_datetime, result_test_name, test_result, .keep_all = TRUE)

#Calculating the timestamp
lab_bef = merge(lab_bef, nac, by = 'id_patient', all.x = TRUE)
lab_bef$Timestamp = as.numeric(difftime(lab_bef$test_result_datetime, 
                                           lab_bef$date_of_birth, units = 'days'))

#Do no run this --------------------------------------------------------------------------------
blood = function(test_name){
  test_name_1 = lab_bef %>% filter(result_test_name==test_name) %>% group_by(id_patient) %>% 
    mutate(test = paste("test", row_number(), sep = "_"))
  test_name_2 <- pivot_wider(test_name_1,id_cols = "id_patient", names_from = "test", 
                          values_from = "test_result")
  return(test_name_2)
}

blood_tests = list()
for (name in unique(lab_bef$result_test_name)) {
  blood_tests[[name]] = blood(name)
}

id_patient = soc10$id_patient
missing = setdiff(id_patient_seq, lab_bef$id_patient)

patient = data.frame(id_patient=missing)

#lab_bef = bind_rows(lab_bef, patient)
#-----------------------------------------------------------------------------------------------
### In long-format with timestamp

blood_long = function(test_name1){
  test_name_1 = lab_bef %>% filter(test_name==test_name1 & !is.na(test_result)) %>% group_by(id_patient)
  test_name_1 = test_name_1[,c(1,15,8)]
  test_name_1 = test_name_1 %>% filter(!is.na(Timestamp))
  #Change if it is necessary
  colnames(test_name_1) <- c('id_patient', paste('Timestamp',test_name1, sep = '_'), paste('test_result',test_name1, sep = '_'))
  return(test_name_1)
}

for (name in unique(lab_bef$test_name)) {
  print(name)
}

blood_tests_long = list()
for (name in unique(lab_bef$test_name)) {
  blood_tests_long[[name]] = blood_long(name)
}

standardize_test_result <- function(df) {
  df[,3] <- scale(df[,3])
  df[,2] <- scale(df[,2])
  return(df)
}

blood_tests_long_stand = lapply(blood_tests_long, standardize_test_result)
View(blood_tests_long_stand$Creatinine)

blood_tests_long_stand$Creatinine %>% ggplot() +
  geom_line(aes(Timestamp_Creatinine, test_result_Creatinine, group = id_patient,
                color = id_patient))

blood_tests_long$Creatinine %>% ggplot() +
  geom_line(aes(Timestamp_Creatinine, test_result_Creatinine, group = id_patient,
                color = id_patient))

for(i in 1:length(blood_tests_long_stand)){
  missing = setdiff(id_patient_seq, blood_tests_long_stand[[i]]$id_patient)
  patient = data.frame(id_patient=missing)
  blood_tests_long_stand[[i]] = bind_rows(blood_tests_long_stand[[i]], patient)
  blood_tests_long_stand[[i]][is.na(blood_tests_long_stand[[i]])]<-0
}

#To verify the data :)
View(blood_tests_long_stand$Creatinine)
#Flta = blood('Folate')

lab_bef %>% filter (test_name == 'Creatinine') %>% ggplot() +
  geom_line(aes(test_result_datetime, test_result, group = id_patient,
                color = id_patient))

#Extracting a static variable

lab_bef1 = lab_bef %>% group_by(id_patient) %>% summarise('number_tests'=n()) 
lab_bef2 = lab_bef %>% filter(test_result_abnormal!='N'&test_result_abnormal!='y') %>%
  group_by(id_patient) %>% summarise('abnormal_test'=n())
lab_bef3 = merge(lab_bef1, lab_bef2, by = 'id_patient', all.x = T)
lab_bef3[is.na(lab_bef3)]<-0
#lab_bef3$abnormal_rate = (lab_bef3$abnormal_test/lab_bef3$number_tests)*100

soc6 = merge(soc5, lab_bef3, by = 'id_patient', all.x = TRUE)
#soc7 = merge(soc6, Dem,  by = 'id_patient', all.x = TRUE)

#4g. Calculate P3 index 

#source("p3index_scoring.R")
#medicine = read_xlsx('List_Medicine_P3_v3.xlsx')
#medicine$Chemical = tolower(medicine$Chemical)
#Med_bef$`Pharmacy List` = tolower(Med_bef$`Pharmacy List`)
#Med_bef1 = merge(Med_bef, medicine, by.x = 'Pharmacy List', by.y = 'Chemical', all.x = T)
#p3 = p3index_scoring(Med_bef1, therapeutic_group_code = 'PHARMAC.codes', 
#                     return_condition_cols = FALSE, id_cols = 'id_patient')

drug = read_xlsx("names_medicine_final - TG.xlsx", sheet = 'names_medicine_final',
                 trim_ws = FALSE)

drug$TG2_C = ifelse(grepl('Diabetes', drug$TG2, ignore.case = TRUE), 'Diabetes',
                    ifelse(drug$TG1 == "Special Foods", "Special Foods", drug$TG2))

drug$TG2_S = ifelse(grepl('Diabetes', drug$TG2, ignore.case = TRUE), 'Diabetes',
                    ifelse(drug$TG1 == "Cardiovascular System", "Cardiovascular", 
                           ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Analgesics" & (drug$TG3 == "Opioid Analgesics"|drug$TG3 == "OPIOIDS"), "Analgesics",
                                  ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Agents for Parkinsonism and Related Disorders", "Agents for Parkinsonism and Related Disorders",
                                         ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Antidepressants", "Antidepressants",
                                                ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Antipsychotic Agents", "Antipsychotics",
                                                       ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Anxiolytics", "Anxiolytics", NA)))))))

Med_bef = merge(Med_bef, drug, by = 'generic_name', all.x = TRUE)
Med_bef = Med_bef %>% filter(!is.na(TG2_C))


Med_bef1 = Med_bef %>% group_by(id_patient, TG2_C) %>% 
  summarise('num_dispense' = n())

Med_bef1 <- pivot_wider(Med_bef1, id_cols = "id_patient", names_from = "TG2_C", 
                         values_from = "num_dispense")
Med_bef1[is.na(Med_bef1)]<-0

soc8 = merge(soc6, Med_bef1,  by = 'id_patient', all.x = TRUE)

#4h. from ARC, extract the total days per service category. 

arc_bef1 = arc_bef %>% group_by(id_patient, `Service.Category`) %>% 
  summarise('Tot_days'=sum(`No..of.Units`, na.rm = T))
arc_bef1 <- pivot_wider(arc_bef1, id_cols = "id_patient", names_from = "Service.Category", 
                         values_from = "Tot_days")
arc_bef1[is.na(arc_bef1)]<-0

soc9 = merge(soc8, arc_bef1,  by = 'id_patient', all.x = TRUE)

#Changing NA for 0
for (names in colnames(soc9[-c(1:11)])) {
  soc9[,names] = ifelse(is.na(soc9[,names]), 0, soc9[,names])
}

soc9$ARC.Facility = ifelse(is.na(soc9$ARC.Facility), 'N', soc9$ARC.Facility)

#soc9$Mortality = ifelse(soc9$date_of_death == '',0,1)

#Age was used for case-control
#soc9$Age = ifelse(soc9$Mortality == 0, trunc((soc9$date_of_birth %--% ymd('2024-02-16'))/years(1)),
#                  trunc(difftime(soc9$date_of_death, soc9$date_of_birth, units = 'days')/365.25))

soc9 <- dummy_cols(soc9, select_columns = "gender_description")

#soc9$Ethnicity = ifelse(soc9$prioritised_ethnicity_description == 'Chinese'|soc9$prioritised_ethnicity_description == 'Indian'|
#                          soc9$prioritised_ethnicity_description == 'Other Asian'| soc9$prioritised_ethnicity_description == 'Southeast Asian', 'Asian',
#                        ifelse(soc9$prioritised_ethnicity_description == 'NZ Maori', 'Māori',
#                               ifelse(soc9$prioritised_ethnicity_description == 'European not further defined'|soc9$prioritised_ethnicity_description == 'NZ European'|
#                                        soc9$prioritised_ethnicity_description == 'Other European', 'NZ European', 
#                                      ifelse(soc9$prioritised_ethnicity_description == 'Middle Eastern'| soc9$prioritised_ethnicity_description == 'Not stated'|
#                                               soc9$prioritised_ethnicity_description == 'Other ethnicity', 'Other', 'Pacific'))))

soc9 <- dummy_cols(soc9, select_columns = "Ethnicity")
soc9 <- dummy_cols(soc9, select_columns = "ARC.Facility")

#soc9_original = read.csv('Static_features.csv', header = T, sep = ',')
#original_col = names(soc9)
#original_col = append(original_col, 'gender_description_Unknown', after = 199)

new_col = names(soc9)

missing_columns <- setdiff(original_col, new_col)

for (col in missing_columns) {
  soc9[[col]] <- 0
}

# Reorder the columns to match the original dataset
soc9 <- soc9[original_col]

missing_columns_1 <- setdiff(new_col, original_col)

soc9 <- soc9[,  !(names(soc9) %in% missing_columns_1)]

for (col in missing_columns) {
  soc9[[col]] <- 0
}

#Export 

write.csv(soc9, 'DifferentTest/Time96m_retrain/Static_features.csv', row.names = FALSE)
write.csv(Visits, 'DifferentTest/Time96m_retrain/Codes_visit1.csv', row.names = FALSE)
write.csv(los, 'DifferentTest/Time96m_retrain/DataPython/los.csv', row.names = FALSE)

#sequences4$assessment_datetime = as.character(sequences4$assessment_datetime)
#write.csv(sequences4, 'DataPython/CAM_scores.csv', row.names = FALSE)
write.csv(CAM_seq3, 'DifferentTest/Time96m_retrain/DataPython/Del_pos.csv', row.names = FALSE)
map2(names(blood_tests_long_stand), blood_tests_long_stand, ~ write.csv(.y, file.path("Retrain/Time36m_retrain/DataPython", paste0(.x, ".csv")), row.names = FALSE))

#write.csv(Dem, 'DEM.csv', row.names = FALSE)

#
#soc10 = merge(soc9, Dem,  by = 'id_patient', all.x = TRUE)
#write.csv(soc10, 'Static_features.csv', row.names = FALSE)

means_df <- soc9[,-c(1:9,11,848:860)] %>%
  group_by(Diagnosis) %>%
  summarize(across(everything(), mean, na.rm = TRUE))

means_df = t(means_df)
col_names <- means_df[1, ]
colnames(means_df) <- col_names
means_df <- means_df[-1, ]

write_xlsx(means_df, 'descriptive_stat.xlsx')

means_df$Dif = as.numeric(means_df$Dementia) - as.numeric(means_df$`No dementia`)

soc_pre = soc_all

AAA = soc_pre %>% group_by(Diagnosis, ARC.Facility) %>% summarise(n()) %>% ungroup() %>%
  group_by(Diagnosis) %>% mutate('pers'= round((`n()`/sum(`n()`))*100, 2))

soc_pre %>% group_by(Diagnosis) %>% summarise('Age' = mean(Age_to_now), 'sd' = sd(Age_to_now))
soc_pre$date_of_death1 = ifelse(soc_pre$date_of_death=='', 0, 1)
soc_pre %>% group_by(Diagnosis) %>% summarise('death'= sum(date_of_death1))

#For the sequences
compute_stats <- function(data, dataset_name) {
  # Filter out patients with timestamp 0
  filtered_data <- subset(data, data[,2]>0)
  
  # Compute number of patients with data
  num_patients_with_data <- n_distinct(filtered_data$id_patient)
  
  # Compute minimum and maximum sequence lengths
  min_sequence_length <- min(table(filtered_data$id_patient))
  med_sequence_length <- median(table(filtered_data$id_patient))
  max_sequence_length <- max(table(filtered_data$id_patient))
  
  # Return results along with dataset name
  return(data.frame(
    dataset_name = dataset_name,
    num_patients_with_data = num_patients_with_data,
    min_sequence_length = min_sequence_length,
    med_sequence_length = med_sequence_length,
    max_sequence_length = max_sequence_length
  ))
}

results_list <- vector("list", length = length(blood_tests_long))


for (i in seq_along(blood_tests_long)) {
  # Compute statistics for the current dataset
  results_list[[i]] <- compute_stats(blood_tests_long[[i]], names(blood_tests_long)[i])
}

result_df1 <- do.call(rbind, results_list)
print(result_df)
write_xlsx(result_df1, 'Retrain/Time96m_retrain/sequential_96m.xlsx')

compute_stats(Visits, "ICD-10")
compute_stats(los, "los")


fil_data = blood_tests_long$Monocytes %>%
  filter(blood_tests_long$Monocytes[,2] != 0)




#Calculating prevalences

#master = read.csv("Full data/master_deidentified.csv", header = TRUE, sep = ",")
#master = master[-137522,]

#master = merge(master, Dem, by = 'id_patient', all.x = TRUE)

master = soc_pre

for (ethnicity in unique(soc_pre$Ethnicity)){
  print(ethnicity)
}

master$date_of_death = dmy(master$date_of_death)
master$year_dementia = year(master$Date_Diagnosis)
master$year_death = year(master$date_of_death)
master$dem_death = ifelse(!is.na(master$date_of_death)&master$Diagnosis=='Dementia', 1, 0)
master$Age = year(dmy('01/01/2024'))-year(ym(master$date_of_birth))

prev = master %>% group_by(year_dementia) %>% summarise(n())
prev1 = master %>% group_by(year_death) %>% summarise(n())
prev2 = merge(prev, prev1, by.x = 'year_dementia', by.y = 'year_death', all.x = TRUE)
colnames(prev2) <- c('year', 'dementia_n', 'death_n')
prev2_1 = master %>% filter(dem_death == 1) %>% group_by(year_death) %>% summarise(n())
colnames(prev2_1) <- c('year_dementia', 'death_dementia_n')
prev2 = merge(prev2, prev2_1, by.x = 'year', by.y = 'year_dementia', all.x = TRUE)

prev2$n_cum <- cumsum(prev2$dementia_n)
prev2 = na.omit(prev2)

n = rep(0, 10)
for (i in 65:74){
  n[i-64] = nrow(subset(master, Age>=i))
}
year=c(2024:2015)
year = sort(year, decreasing = T)
n = cbind(year, n)

prev2 = merge(prev2, n, by = 'year',all.x = TRUE)


prev2$cum_death = cumsum(prev2$death_n)
prev2$cum_death_dem = cumsum(prev2$death_dementia_n)

#prev2$n_no_death = prev2$n-prev2$cum_death
prev2$inc = ((prev2$dementia_n)/(prev2$n - prev2$death_n))*1000#(prev2$dementia_n-prev2$death_dementia_n)/prev2$n_no_death
prev2$prev = (prev2$n_cum - prev2$cum_death_dem)/(prev2$n-prev2$cum_death) #(prev2$n_cum-prev2$cum_death_dem)/(prev2$n_no_death-prev2$cum_death)
prev2$Ethnicity = "All"

prev_ethnicity = function(Ethni, data = soc_pre){
  
  master1 = soc_pre %>% filter(Ethnicity == Ethni)
  
  master1$date_of_death = dmy(master1$date_of_death)
  master1$year_dementia = year(master1$Date_Diagnosis)
  master1$year_death = year(master1$date_of_death)
  master1$dem_death = ifelse(!is.na(master1$date_of_death)&master1$Diagnosis=='Dementia', 1, 0)
  master1$Age = year(dmy('01/01/2024'))-year(ym(master1$date_of_birth))
  
  prev = master1 %>% group_by(year_dementia) %>% summarise(n())
  prev1 = master1 %>% group_by(year_death) %>% summarise(n())
  prev2 = merge(prev, prev1, by.x = 'year_dementia', by.y = 'year_death', all.x = TRUE)
  colnames(prev2) <- c('year', 'dementia_n', 'death_n')
  prev2_1 = master1 %>% filter(dem_death == 1) %>% group_by(year_death) %>% summarise(n())
  colnames(prev2_1) <- c('year_dementia', 'death_dementia_n')
  prev2 = merge(prev2, prev2_1, by.x = 'year', by.y = 'year_dementia', all.x = TRUE)
  
  prev2$n_cum <- cumsum(prev2$dementia_n)
  prev2 = na.omit(prev2)
  
  n = rep(0, 10)
  for (i in 65:74){
    n[i-64] = nrow(subset(master1, Age>=i))
  }
  year=c(2024:2015)
  year = sort(year, decreasing = T)
  n = cbind(year, n)
  
  prev2 = merge(prev2, n, by = 'year',all.x = TRUE)
  
  prev2$cum_death = cumsum(prev2$death_n)
  prev2$cum_death_dem = cumsum(prev2$death_dementia_n)
  
  #prev2$n_no_death = prev2$n-prev2$cum_death
  prev2$inc = ((prev2$dementia_n)/(prev2$n - prev2$death_n))*1000#(prev2$dementia_n-prev2$death_dementia_n)/prev2$n_no_death
  prev2$prev = (prev2$n_cum - prev2$cum_death_dem)/(prev2$n-prev2$cum_death) #(prev2$n_cum-prev2$cum_death_dem)/(prev2$n_no_death-prev2$cum_death)
  prev2$Ethnicity = Ethni
  
  return(prev2)
}

Prev_Asian = prev_ethnicity(Ethni = "Asian", data = soc_pre)
Prev_European = prev_ethnicity(Ethni = "European", data = soc_pre)
Prev_Maori = prev_ethnicity(Ethni = "Māori", data = soc_pre)
Prev_Pacific = prev_ethnicity(Ethni = "Pacific", data = soc_pre)

Prev_all = rbind(prev2, Prev_Asian, Prev_Maori, Prev_Pacific, Prev_European)
Prev_all = Prev_all %>% filter(!year %in% c(2015, 2016, 2024))

ggplot(Prev_all, aes(year, prev, group = Ethnicity,
                  color = Ethnicity)) + geom_line() + geom_point(aes(year, prev, group = Ethnicity,
                                                                     color = Ethnicity),size=2) 
ggsave("Prev.pdf", width = 9, height = 5, dpi = 600)

ggplot(Prev_all, aes(year, inc, group = Ethnicity,
                     color = Ethnicity)) + geom_line() + geom_point(aes(year, inc, group = Ethnicity,
                                                                        color = Ethnicity),size=2)
ggsave("Inc.pdf", width = 9, height = 5, dpi = 600)

prev2$inc = (prev2$dementia_n)/(prev2$n)#(prev2$dementia_n-prev2$death_dementia_n)/prev2$n_no_death
prev2$prev = (prev2$n_cum)/prev2$n #(prev2$n_cum-prev2$cum_death_dem)/(prev2$n_no_death-prev2$cum_death)


prev3 = melt(prev2, id.vars=c("year","dementia_n","death_n","n","n_cum","n_no_death", "cum_death", 'cum_death_dem',
                              'death_dementia_n'))

prev3$se = NA
for (i in 1:nrow(prev3)){
  if(prev3$variable[i] == 'inc'){
    prev3$se[i] = 1.96*sqrt((prev3$value[i]*(1-prev3$value[i]))/prev3$n[i])#n_no_death[i])
  } else if (prev3$variable[i] == 'prev'){
    prev3$se[i] = 1.96*sqrt((prev3$value[i]*(1-prev3$value[i]))/(prev3$n[i]))#n_no_death[i]-prev3$cum_death[i]))
  } else {
    0
  }
}

ggplot(prev3, aes(year, value, group = variable,
                  color = variable)) +
  geom_line() + geom_point(aes(year, value, group = variable,
                                                    color = variable),size=2) + geom_errorbar(aes(ymin=value-se, ymax=value+se), width=.1, 
                                                                                              position=position_dodge(0.05)) + ylab('') + xlab('Year')

ggsave("Prev3.pdf", width = 9, height = 5, dpi = 600)


prev_ethn_age = function(ethni, year, data = soc_pre){
  
master1 = data %>% filter(Ethnicity == ethni)

master1$date_of_death = dmy(master1$date_of_death)
master1$year_dementia = year(master1$Date_Diagnosis)
master1$year_death = year(master1$date_of_death)

#master1 = master1 %>% filter((year_dementia <= 2023 | year_death <= 2023)|(is.na(year_dementia)|is.na(year_death)))

master1$dem_death = ifelse(!is.na(master1$date_of_death)&master1$Diagnosis=='Dementia', 1, 0)
  
#master1$Age2024 = year(ymd('2024-03-19'))-year(ym(master1$date_of_birth))
master1$Age_group = cut(master1$Age_to_now-(2024-year), c(0, 64, 74, 84, 94, 120))
  
#master1$Age2023 = year(ymd('2023-12-31'))-year(ym(master1$date_of_birth))
#master1$Age_group_23 = cut(master1$Age2023, c(0, 64, 74, 84, 94, 120))
  
prev = master1 %>% filter(Diagnosis == 'Dementia' & year_dementia <= year) %>% group_by(Age_group) %>% summarise('Dementia' = n())
prev1 = master1 %>% filter(!is.na(date_of_death) & year_death <= year) %>% group_by(Age_group) %>% summarise('Death' = n())
prev2 = merge(prev, prev1, by = 'Age_group', all.x = TRUE)

#colnames(prev2) <- c('year', 'dementia_n', 'death_n')

prev2_1 = master1 %>% filter(dem_death == 1 & year_death <= year) %>% group_by(Age_group) %>% summarise('Death_Dem' = n())
#colnames(prev2_1) <- c('year_dementia', 'death_dementia_n')
prev2 = merge(prev2, prev2_1, by = 'Age_group', all.x = TRUE)

prev2_2 = master1 %>% filter(Age_to_now >= 65-(2024-year)) %>% group_by(Age_group) %>% summarise('n'=n())

prev2 = merge(prev2, prev2_2, by = 'Age_group', all.x = TRUE)
  
#prev2$n_cum <- cumsum(prev2$dementia_n)
#prev2 = na.omit(prev2)
  
#n = rep(0, 10)
#for (i in 65:74){
#n[i-64] = nrow(subset(master1, Age>=i))
#}
#year=c(2024:2015)
#year = sort(year, decreasing = T)
#n = cbind(year, n)
#prev2 = merge(prev2, n, by = 'year',all.x = TRUE)
  
#prev2$cum_death = cumsum(prev2$death_n)
#prev2$cum_death_dem = cumsum(prev2$death_dementia_n)
  
#prev2$n_no_death = prev2$n-prev2$cum_death
#prev2$inc = ((prev2$dementia_n)/(prev2$n - prev2$death_n))*1000#(prev2$dementia_n-prev2$death_dementia_n)/prev2$n_no_death
prev2$prev = (prev2$Dementia - prev2$Death_Dem)/(prev2$n-prev2$Death) #(prev2$n_cum-prev2$cum_death_dem)/(prev2$n_no_death-prev2$cum_death)
prev2$Year = year
prev2$Ethnicity = ethni
  
  return(prev2)
}

prev_tot = list()

for(eth in unique(soc_pre$Ethnicity)){
  for (i in 2015:2024){
    dat = prev_ethn_age(eth, i, data = soc_pre)
    prev_tot[[paste(eth, i, sep = "_")]] <- dat
  }
}

final_prev <- do.call(rbind, prev_tot)
final_prev = final_prev %>% filter(Age_group != "(0,64]" & Ethnicity %in% c("European", "Pacific", 'Māori', 'Asian'))

final_prev1 = pivot_wider(final_prev, id_cols = c("Age_group", 'Ethnicity'), names_from = "Year", 
                             values_from = "prev")

for (i in 3:12) {
  final_prev1[,i] = round(final_prev1[,i]*100,2)
}

library(xtable)
xtable(final_prev1)



library(FactoClass)

acp = dudi.pca(Med_bef1[,-1])
plot(acp)

inertia.dudi(acp)

acp1 = cbind(Med_bef1$id_patient, acp$l1)
colnames(acp1) <- c('id_patient', 'CP1', 'CP2')
acp1 = merge(soc, acp1, by = 'id_patient', all.x = T)

acp1 %>% group_by(Diagnosis) %>% summarise(mean(CP2, na.rm = T))

#Describe the whole datasets

final_id = soc_pre$id_patient

IPA_f = IPA %>% filter(id_patient %in% final_id)
AAA = IPA_f %>% group_by(id_patient) %>% summarise(n())
AAA = IPA_f %>% group_by(id_patient, clinical_code) %>% summarise(n())

AAA1 = AAA %>% group_by(id_patient) %>% summarise(n())

IPA_f$Dif_ad_di = as.numeric(difftime(IPA_f$discharge_datetime, IPA_f$admission_datetime, units = 'days'))
IPA_f1 = IPA_f %>% filter(Dif_ad_di>=1 & coding_sequence_number == 1)

#Sequence of length of stay
nc = soc_pre[,c(1:2,10)]
nc$date_of_birth = ym(nc$date_of_birth)

IPA_f1 = merge(IPA_f1, nc, by = 'id_patient', all.x = TRUE)
IPA_f1$Timestamp = as.numeric(difftime(IPA_f1$admission_datetime, 
                                         IPA_f1$date_of_birth, units = 'days'))
los_t = IPA_f1[,c(1,16,14,17)]
los_t = los_t %>% filter(!is.na(Timestamp))

mean_data <- los_t %>%
  group_by(Diagnosis, Timestamp) %>%
  summarize(Mean_los = mean(Dif_ad_di, na.rm = TRUE), .groups = "drop")

mean_data %>% ggplot(aes(Timestamp, Mean_los, group = Diagnosis,
                   color = Diagnosis)) + geom_line()


st1 = IPA_f1 %>% group_by(id_patient) %>% summarise('number_admissions'= n())
nc1 = merge(nc, st1, by='id_patient', all.x = TRUE)
nc1$number_admissions = ifelse(is.na(nc1$number_admissions), 0, nc1$number_admissions)
nc1 %>% group_by(Diagnosis) %>% summarise(mean(number_admissions), sd(number_admissions))
t.test(number_admissions~Diagnosis, data=nc1)

EDA_f1 = EDA %>% group_by(id_patient) %>% summarise('number_ED_attendance'=n())
nc2 = merge(nc, EDA_f1, by='id_patient', all.x = TRUE)
nc2$number_ED_attendance = ifelse(is.na(nc2$number_ED_attendance), 0, nc2$number_ED_attendance)
nc2 %>% group_by(Diagnosis) %>% summarise(mean(number_ED_attendance), sd(number_ED_attendance))
t.test(number_ED_attendance~Diagnosis, data=nc2)

OPA$outpatient_specialty_desc = ifelse(OPA$outpatient_specialty_desc == 'Maxillo Facial Surgery', 'Maxillo-Facial Surgery',
                                       ifelse(OPA$outpatient_specialty_desc == 'Public Health Nursing', 'EC Nursing',OPA$outpatient_specialty_desc))
OPA_f1 = OPA %>% group_by(id_patient, outpatient_specialty_desc) %>%
  summarise("Number_appoint"=n())
AAA = OPA_f1 %>% group_by(outpatient_specialty_desc) %>% summarise(n())
OPA_f2 <- pivot_wider(OPA_f1,id_cols = "id_patient", names_from = "outpatient_specialty_desc", 
                        values_from = "Number_appoint")
nc3 = merge(nc, OPA_f2, by='id_patient', all.x = TRUE)
nc3[is.na(nc3)]<-0

for (i in 4:60) {
  print(colnames(nc3)[i])
  print(t.test(nc3[,i]~Diagnosis, data = nc3))
}

OPA$attendance_status = ifelse(OPA$attendance_status_desc=='Did not wait'|
                                     OPA$attendance_status_desc=='Not Specified'|
                                     OPA$attendance_status_desc=='Service Not Delivered/Incomplete','Other',
                                   OPA$attendance_status_desc)

OPA_f3 = OPA %>% group_by(id_patient, attendance_status) %>%
  summarise("Number_attendance"=n())

OPA_f4 <- pivot_wider(OPA_f3,id_cols = "id_patient", names_from = "attendance_status", 
                        values_from = "Number_attendance")
nc4 = merge(nc, OPA_f4, by='id_patient', all.x = TRUE)
nc4[is.na(nc4)]<-0

for (i in 4:7) {
  print(colnames(nc4)[i])
  print(t.test(nc4[,i]~Diagnosis, data = nc4))
}

cont_f1 = cont %>% group_by(id_patient, contact_status_desc) %>% summarise("Contact_status"=n())
cont_f1 <- pivot_wider(cont_f1,id_cols = "id_patient", names_from = "contact_status_desc", 
                         values_from = "Contact_status")
nc5 = merge(nc, cont_f1, by='id_patient', all.x = TRUE)
nc5[is.na(nc5)]<-0
for (i in 4:6) {
  print(colnames(nc5)[i])
  print(t.test(nc5[,i]~Diagnosis, data = nc5))
}


CAM_f1 = CAM %>% group_by(id_patient) %>% 
  summarise('number_CAM'=n(),'number_admissions_cam'=n_distinct(episode_number)) 

CAM_f2 = CAM %>% filter(CAM_Score_3._Evidence_Focus=='Yes') %>%
  group_by(id_patient) %>% summarise('Number_positive_delirium' = n())

CAM_f2 = merge(CAM_f1, CAM_f2, by = 'id_patient', all.x = T)
CAM_f2[is.na(CAM_f2)]<-0
nc6 = merge(nc, CAM_f2, by='id_patient', all.x = TRUE)
nc6[is.na(nc6)]<-0

for (i in 4:6) {
  print(colnames(nc6)[i])
  print(t.test(nc6[,i]~Diagnosis, data = nc6))
}

###Sequence with timestamp
CAM_f = merge(CAM, nc, by = 'id_patient', all.x = TRUE)
CAM_f$Timestamp = as.numeric(difftime(CAM_f$assessment_datetime, 
                                        CAM_f$date_of_birth, units = 'days'))

CAM_sq = CAM_f %>% group_by(id_patient, episode_number) %>% 
  mutate('delirium_pos'=if_else(total_score>=3 & CAM_Score_3._Evidence_Focus == 'Yes', 1, 0))

CAM_sq1 = CAM_sq %>% group_by(id_patient, episode_number) %>% 
  summarise('num_del'=sum(delirium_pos))

CAM_sq2 = CAM_sq %>% group_by(id_patient, episode_number) %>% 
  summarise('Timestamp'=min(Timestamp))

CAM_sq3 = merge(CAM_sq1, CAM_sq2, by = c('id_patient', 'episode_number'))

CAM_sq3 = CAM_sq3[,c(1,4,3)]
CAM_sq3 = CAM_sq3 %>% filter(!is.na(Timestamp))
CAM_sq3 = merge(CAM_sq3, nc, by='id_patient', all.x = TRUE)

CAM_sq3 %>% ggplot(aes(Timestamp, num_del, group = id_patient,
                         color = Diagnosis)) + geom_line()

mean_data1 <- CAM_sq3 %>%
  group_by(Diagnosis, Timestamp) %>%
  summarize(Mean_del = mean(num_del, na.rm = TRUE), .groups = "drop")

mean_data1 %>% ggplot(aes(Timestamp, Mean_del, group = Diagnosis,
                         color = Diagnosis)) + geom_line() 

lab$test_result = as.numeric(gsub("[<>]","",lab$test_result))
lab = lab %>% filter(!is.na(test_result))

lab = lab %>% distinct(id_patient, test_result_datetime, result_test_name, test_result, .keep_all = TRUE)

lab_f1 = lab %>% group_by(id_patient) %>% summarise('number_tests'=n()) 
lab_f2 = lab %>% filter(test_result_abnormal!='N'&test_result_abnormal!='y') %>%
  group_by(id_patient) %>% summarise('abnormal_test'=n())
lab_f3 = merge(lab_f1, lab_f2, by = 'id_patient', all.x = T)
lab_f3[is.na(lab_f3)]<-0

nc7 = merge(nc, lab_f3, by='id_patient', all.x = TRUE)
nc7[is.na(nc7)]<-0

for (i in 4:5) {
  print(colnames(nc7)[i])
  print(t.test(nc7[,i]~Diagnosis, data = nc7))
}

lab_f = merge(lab, nc, by = 'id_patient', all.x = TRUE)
lab_f$Timestamp = as.numeric(difftime(lab_f$test_result_datetime, 
                                        lab_f$date_of_birth, units = 'days'))

### In long-format with timestamp

blood_long_f = function(test_name1){
  test_name_1 = lab_f %>% filter(test_name==test_name1 & !is.na(test_result)) %>% group_by(id_patient)
  test_name_1 = test_name_1[,c(1,12:13,8)]
  test_name_1 = test_name_1 %>% filter(!is.na(Timestamp))
  #Change if it is necessary
  colnames(test_name_1) <- c('id_patient', 'Diagnosis', paste('Timestamp',test_name1, sep = '_'), paste('test_result',test_name1, sep = '_'))
  return(test_name_1)
}

blood_tests_tot = list()
for (name in unique(lab_f$test_name)) {
  blood_tests_tot[[name]] = blood_long_f(name)
}

blood_tests_tot$Haematocrit %>% ggplot() +
  geom_line(aes(Timestamp_Haematocrit, test_result_Haematocrit, group = id_patient,
                color = Diagnosis))

mean_data1 <- blood_tests_tot$HbA1c %>%
  group_by(Diagnosis, Timestamp_HbA1c) %>%
  summarize(Mean = median(test_result_HbA1c, na.rm = TRUE), .groups = "drop")

mean_data1 %>% ggplot() +
  geom_line(aes(Timestamp_HbA1c, Mean, group = Diagnosis,
                color = Diagnosis))

compute_stats_t <- function(data, dataset_name) {
  # Filter out patients with timestamp 0
  #filtered_data <- subset(data, data[,2]>0)
  
  # Compute number of patients with data
  num_patients_with_data <- n_distinct(data$id_patient)
  
  # Compute minimum and maximum sequence lengths
  min_sequence_length <- min(table(data$id_patient))
  med_sequence_length <- median(table(data$id_patient))
  max_sequence_length <- max(table(data$id_patient))
  
  # Return results along with dataset name
  return(data.frame(
    dataset_name = dataset_name,
    num_patients_with_data = num_patients_with_data,
    min_sequence_length = min_sequence_length,
    med_sequence_length = med_sequence_length,
    max_sequence_length = max_sequence_length
  ))
}

results_list_t <- vector("list", length = length(blood_tests_tot))

for (i in seq_along(blood_tests_tot)) {
  # Compute statistics for the current dataset
  results_list_t[[i]] <- compute_stats_t(blood_tests_tot[[i]], names(blood_tests_tot)[i])
}

result_df <- do.call(rbind, results_list_t)
xtable(result_df)

Med_f = merge(Med, drug, by = 'generic_name', all.x = TRUE)
Med_f = Med_f %>% filter(!is.na(TG2_C))


Med_f1 = Med_f %>% group_by(id_patient, TG2_C) %>% 
  summarise('num_dispense' = n())

Med_f1 <- pivot_wider(Med_f1, id_cols = "id_patient", names_from = "TG2_C", 
                        values_from = "num_dispense")
Med_f1[is.na(Med_f1)]<-0

nc8 = merge(nc, Med_f1, by='id_patient', all.x = TRUE)
nc8[is.na(nc8)]<-0

BBB = matrix(rep(0, 128*4), ncol = 4)

for (i in 1:128) {
  BBB[i,1] = colnames(nc8)[i+3]
  BBB[i,2] = mean(nc8[nc8$Diagnosis=='Dementia',i+3])
  BBB[i,3] = mean(nc8[nc8$Diagnosis=='No dementia',i+3])
  BBB[i,4] = t.test(nc8[,i+3]~Diagnosis, data = nc8)$p.value
}

BBB = as.data.frame(BBB)
colnames(BBB)<-c('TG2', 'Dementia', 'No dementia', 'p')
BBB$Dementia = as.numeric(BBB$Dementia)
BBB$`No dementia` = as.numeric(BBB$`No dementia`)
BBB$p = as.numeric(BBB$p)
BBB$Dementia = round(BBB$Dementia,2)
BBB$`No dementia` = round(BBB$`No dementia`,2)
BBB$p = round(BBB$p,4)
BBB = BBB[order(BBB[[1]]),]

xtable(BBB)

arc_f1 = arc %>% group_by(id_patient, `Service.Category`) %>% 
  summarise('Tot_days'=sum(`No..of.Units`, na.rm = T))
arc_f1 <- pivot_wider(arc_f1, id_cols = "id_patient", names_from = "Service.Category", 
                        values_from = "Tot_days")
arc_f1[is.na(arc_f1)]<-0

nc9 = merge(nc, arc_f1, by='id_patient', all.x = TRUE)
nc9[is.na(nc9)]<-0

for (i in 4:10) {
  print(colnames(nc9)[i])
  print(t.test(nc9[,i]~Diagnosis, data = nc9))
}


#####
soc %>% group_by(Diagnosis, Ethnicity) %>% summarise(n())

soc %>% group_by(Diagnosis) %>% summarise('death'= sum(date_of_death!=''))

#Summer Data

summer = read.delim('clipboard', header = T)
summer = summer %>% filter(!is.na(id_patient))

soc_sum = read.csv("Full data/master_deidentified.csv", header = TRUE, sep = ",")

soc_sum = soc_sum %>% filter(id_patient %in% summer$id_patient)
IPA_sum = IPA %>% filter(id_patient %in% summer$id_patient)
EDA_sum = EDA %>% filter(id_patient %in% summer$id_patient)
OPA_sum = OPA %>% filter(id_patient %in% summer$id_patient)
cont_sum = cont %>% filter(id_patient %in% summer$id_patient)
CAM_sum = CAM %>% filter(id_patient %in% summer$id_patient)
inter_sum = inter %>% filter(id_patient %in% summer$id_patient)
lab_sum = lab %>% filter(id_patient %in% summer$id_patient)
Med_sum = Med %>% filter(id_patient %in% summer$id_patient)
arc_sum = arc %>% filter(id_patient %in% summer$id_patient)

CT_rep = read.delim('clipboard', header = TRUE)

CT_sum = CT_rep %>% filter(database_number %in% summer$database_number)

tables = list(summer, soc_sum, IPA_sum, EDA_sum, OPA_sum, cont_sum, 
              CAM_sum, inter_sum, lab_sum, Med_sum, arc_sum, 
              CT_sum)
library(openxlsx)
wb <- createWorkbook()

for (i in seq_along(tables)) {
  addWorksheet(wb, paste("Sheet", i))            # Name each sheet as "Sheet 1", "Sheet 2", etc.
  writeData(wb, sheet = i, tables[[i]])          # Write each table to a different sheet
}

# Save the workbook as an Excel file
saveWorkbook(wb, "tables.xlsx", overwrite = TRUE)

# Extracting data for the final survival model

keep_before_dem = function(original_set, dementia_date, var_date_name, days_bef = 0){
  dat1 = merge(original_set, dementia_date, by = 'id_patient', all.y = T)
  dat1[,var_date_name] = ymd_hms(dat1[,var_date_name], truncated = 3)
  dat1$Date_Diagnosis = ymd_hms(dat1$Date_Diagnosis, truncated = 3)
  dat1$Diff = difftime(dat1$Date_Diagnosis, dat1[,var_date_name], units = 'days')
  dat2 = dat1 %>% filter(Diff>days_bef)#|is.na(Diff))
  return(dat2)
}

IPA_bef = keep_before(IPA, Dem_NoDem, "admission_datetime", days_bef = 1095)
EDA_bef = keep_before(EDA, Dem_NoDem, "arrived_datetime", days_bef = 1095)
OPA_bef = keep_before(OPA, Dem_NoDem, "appointment_start_date", days_bef = 1095)
cont_bef = keep_before(cont, Dem_NoDem, "contact_start_datetime", days_bef = 1095)
CAM_bef = keep_before(CAM, Dem_NoDem, "assessment_datetime", days_bef = 1095)
inter_bef = keep_before(inter, Dem_NoDem, "DateSigned")
lab_bef = keep_before(lab, Dem_NoDem, "test_result_datetime", days_bef = 1095)
Med_bef = keep_before(Med, Dem_NoDem, "dispense_date", days_bef = 1095)
arc_bef = keep_before(arc, Dem_NoDem, "Service.Start", days_bef = 1095)

surv1 = master[,c(1,3,4,7,9,10,11,17)]

IPA_bef$Dif_ad_di = as.numeric(difftime(IPA_bef$discharge_datetime, IPA_bef$admission_datetime, units = 'days'))
IPA_bef1 = IPA_bef %>% filter(Dif_ad_di>=1 & coding_sequence_number == 1)
stat1 = IPA_bef1 %>% group_by(id_patient) %>% summarise('number_admissions'= n())

soc1 = merge(soc, stat1, by = 'id_patient', all.x = TRUE)
#soc1 = merge(soc_test1, stat1, by = 'id_patient', all.x = TRUE) #Only test
#soc1$tot_los = ifelse(is.na(soc1$tot_los),0,soc1$tot_los)
soc1$number_admissions = ifelse(is.na(soc1$number_admissions),0,soc1$number_admissions)

#Sequence of length of stay
nac = soc[,1:2]
nac$date_of_birth = ym(nac$date_of_birth)

#FOR TEST+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
nac = soc_test1[,1:2]
nac$date_of_birth = ym(nac$date_of_birth)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

IPA_bef1 = merge(IPA_bef1, nac, by = 'id_patient', all.x = TRUE)
IPA_bef1$Timestamp = as.numeric(difftime(IPA_bef1$admission_datetime, 
                                         IPA_bef1$date_of_birth, units = 'days'))
los = IPA_bef1[,c(1,19,17)]
los = los %>% filter(!is.na(Timestamp))
los$Dif_ad_di = scale(los$Dif_ad_di)
los$Timestamp = scale(los$Timestamp)

los %>% ggplot(aes(Timestamp, Dif_ad_di, group = id_patient,
                   color = id_patient)) + geom_line()

missing = setdiff(id_patient_seq, los$id_patient)
patient = data.frame(id_patient=missing)
los = bind_rows(los, patient)
los[is.na(los)]<-0

#Sequence of ICD-10 codes for each visit (Longitudinal)
#icd10 = read_xlsx("ICD10_Vocabulary.xlsx") #To change with the new data

#icd10_train <- IPA_bef %>%
#  distinct(clinical_code) %>%
#  mutate(Number_ICD10 = row_number())

IPA_bef = merge(IPA_bef, icd10_train, by = 'clinical_code', all.x = TRUE)

# Do not run this ------------------------------------------------------------------------
sequences <- IPA_bef %>% filter(Dif_ad_di>=1) %>%
  group_by(id_patient, admission_datetime) %>%
  summarize(ICD_Sequence = list(clinical_code)) %>%
  mutate(visit_number = paste("visit",row_number(), sep = "_"))

sequences1 <- pivot_wider(sequences,id_cols = "id_patient", names_from = "visit_number", 
                          values_from = "ICD_Sequence")

####Another format just in case is needed
sequences3 <- IPA_bef %>% filter(Dif_ad_di>=1) %>%
  group_by(id_patient, admission_datetime) %>%
  summarize(ICD_Sequence = list(Number_ICD10)) 

for (i in 1:nrow(sequences3)) {
  sequences3$ICD_Sequence[i] <- paste(unlist(sequences3$ICD_Sequence[i]), collapse = " ")
}
sequences3$ICD_Sequence = as.character(sequences3$ICD_Sequence)

sequences3 = merge(sequences3, nac, by = 'id_patient', all.x = TRUE)
sequences3$Timestamp = as.numeric(difftime(sequences3$admission_datetime, 
                                           sequences3$date_of_birth, units = 'days'))
sequences3 = sequences3[,c(1,5,3)]
#------------------------------------------------------------------------------------------

#Another way 
#IPA_bef$admission_datetime = ymd_hms(IPA_bef$admission_datetime)
#IPA_bef$coded_date = ymd_hms(IPA_bef$coded_date)

IPA_bef$coded_date = ifelse(!is.na(IPA_bef$coded_date),IPA_bef$coded_date,IPA_bef$admission_datetime)
IPA_bef$coded_date = as.POSIXct(IPA_bef$coded_date, tz=Sys.timezone())
IPA_bef = merge(IPA_bef, nac, by = 'id_patient', all.x = TRUE)
IPA_bef$Timestamp = as.numeric(difftime(IPA_bef$coded_date, 
                                        IPA_bef$date_of_birth, units = 'days'))
Visits = IPA_bef %>% filter(Dif_ad_di>=1)
Visits = Visits[,c(1,20,18)]
Visits = Visits %>% filter(!is.na(Timestamp))
Visits = Visits %>% filter(!is.na(Number_ICD10))
Visits$Timestamp = scale(Visits$Timestamp)

missing = setdiff(id_patient_seq, Visits$id_patient)
patient = data.frame(id_patient=missing)
Visits = bind_rows(Visits, patient)
Visits[is.na(Visits)]<-0 #Ask about the missing data


#4b. For ED, extract number of time in ED before dementia. 
EDA_bef1 = EDA_bef %>% group_by(id_patient) %>% summarise('number_ED_attendance'=n())

soc2 = merge(soc1, EDA_bef1, by = 'id_patient', all.x = TRUE)
soc2$number_ED_attendance = ifelse(is.na(soc2$number_ED_attendance),0,soc2$number_ED_attendance)

#4c. For OP, extract the number of appointments by specialty
OPA_bef1 = OPA_bef %>% group_by(id_patient, outpatient_specialty_desc) %>%
  summarise("Number_appoint"=n())

OPA_bef$attendance_status = ifelse(OPA_bef$attendance_status_desc=='Did not wait'|
                                     OPA_bef$attendance_status_desc=='Not Specified'|
                                     OPA_bef$attendance_status_desc=='Service Not Delivered/Incomplete','Other',
                                   OPA_bef$attendance_status_desc)

OPA_bef3 = OPA_bef %>% group_by(id_patient, attendance_status) %>%
  summarise("Number_attendance"=n())

OPA_bef2 <- pivot_wider(OPA_bef1,id_cols = "id_patient", names_from = "outpatient_specialty_desc", 
                        values_from = "Number_appoint")

OPA_bef4 <- pivot_wider(OPA_bef3,id_cols = "id_patient", names_from = "attendance_status", 
                        values_from = "Number_attendance")

OPA_bef5 = merge(OPA_bef2, OPA_bef4, by='id_patient', all = T)
colnames(OPA_bef5) <- c('id_patient',paste('OP',colnames(OPA_bef5)[-1],sep="_"))
OPA_bef5[is.na(OPA_bef5)]<-0

soc3 = merge(soc2, OPA_bef5, by = 'id_patient', all.x = TRUE)

#4d. For Contact, extract number of contact occurred, planned or cancelled. 
cont_bef1 = cont_bef %>% group_by(id_patient, contact_status_desc) %>% summarise("Contact_status"=n())
cont_bef1 <- pivot_wider(cont_bef1,id_cols = "id_patient", names_from = "contact_status_desc", 
                         values_from = "Contact_status")
colnames(cont_bef1) <- c('id_patient',paste('Contact',colnames(cont_bef1)[-1],sep="_"))
cont_bef1[is.na(cont_bef1)]<-0

soc4 = merge(soc3, cont_bef1, by = 'id_patient', all.x = TRUE)

#4e. For CAM, extract number of assessments, number of delirium screened, number non-consecutive delirium 

CAM_bef1 = CAM_bef %>% group_by(id_patient) %>% 
  summarise('number_CAM'=n(),'number_admissions_cam'=n_distinct(episode_number)) 

CAM_bef2 = CAM_bef %>% filter(CAM_Score_3._Evidence_Focus=='Yes') %>%
  group_by(id_patient) %>% summarise('Number_positive_delirium' = n())

CAM_bef2 = merge(CAM_bef1, CAM_bef2, by = 'id_patient', all.x = T)
CAM_bef2[is.na(CAM_bef2)]<-0

soc5 = merge(soc4, CAM_bef2, by = 'id_patient', all.x = TRUE)

# Do no run this 
#--------------------------------------------------------------------------------------------------
#Calculating the timestamp
CAM_bef = CAM_bef %>% group_by(id_patient) %>% mutate(CAM = paste("CAM", row_number(), sep = "_"))
sequence2 <- pivot_wider(CAM_bef,id_cols = "id_patient", names_from = "CAM", 
                         values_from = "total_score")
#--------------------------------------------------------------------------------------------------

###Sequence with timestamp
CAM_bef = merge(CAM_bef, nac, by = 'id_patient', all.x = TRUE)
CAM_bef$Timestamp = as.numeric(difftime(CAM_bef$assessment_datetime, 
                                        CAM_bef$date_of_birth, units = 'days'))

CAM_seq = CAM_bef %>% group_by(id_patient, episode_number) %>% 
  mutate('delirium_pos'=if_else(total_score>=3 & CAM_Score_3._Evidence_Focus == 'Yes', 1, 0))

CAM_seq1 = CAM_seq %>% group_by(id_patient, episode_number) %>% 
  summarise('num_del'=sum(delirium_pos))

CAM_seq2 = CAM_seq %>% group_by(id_patient, episode_number) %>% 
  summarise('Timestamp'=min(Timestamp))

CAM_seq3 = merge(CAM_seq1, CAM_seq2, by = c('id_patient', 'episode_number'))

CAM_seq3 = CAM_seq3[,c(1,4,3)]
CAM_seq3 = CAM_seq3 %>% filter(!is.na(Timestamp))
CAM_seq3$num_del = scale(CAM_seq3$num_del)
CAM_seq3$Timestamp = scale(CAM_seq3$Timestamp)

#sequences4 = CAM_bef[,c(1,16,10)]

missing = setdiff(id_patient_seq, CAM_seq3$id_patient)
patient = data.frame(id_patient=missing)
CAM_seq3 = bind_rows(CAM_seq3, patient)
CAM_seq3[is.na(CAM_seq3)]<-0 #Ask about the missing data

#4f. For blood tests, extract all sequences. First remove those rows when the test result is
# not numeric. 

lab_bef$test_result = as.numeric(gsub("[<>]","",lab_bef$test_result))
lab_bef = lab_bef %>% filter(!is.na(test_result))

# Make the test names consistent 

#lab_bef$result_test_name = ifelse(lab_bef$result_test_name == "WBC - White Cell Count","WBC",
#                                  ifelse(lab_bef$result_test_name == 'RBC - Red Cell Count',"RBC",
#                                         ifelse(lab_bef$result_test_name == 'Nucleated RBCs'|lab_bef$result_test_name == 'Nucleated Red Cell Count'|lab_bef$result_test_name == 'NRBCS'|lab_bef$result_test_name == 'Nucleated RBC','RBC (nucleated)',
#                                                ifelse(lab_bef$result_test_name == 'Platelet Count','Platelets',
#                                                       ifelse(lab_bef$result_test_name == 'Mean Cell Volume'|lab_bef$result_test_name == 'MCV - Mean Cell Volume','MCV',
#                                                              ifelse(lab_bef$result_test_name == 'Mean Cell Haemoglobin'|lab_bef$result_test_name == 'MCH - Mean Cell Haemoglobin','MCH',
#                                                                     ifelse(lab_bef$result_test_name == 'Hct - Haematocrit'|lab_bef$result_test_name == "HCT",'Haematocrit',
#                                                                            ifelse(lab_bef$result_test_name == "Hb  - Haemoglobin", 'Haemoglobin', 
#                                                                                   ifelse(lab_bef$result_test_name == 'B12', "Vitamin B12", 
#                                                                                          ifelse(lab_bef$result_test_name == 'Free T3','T3 (free)',
#                                                                                                 ifelse(lab_bef$result_test_name == 'Free T4', 'T4 (free)',
#                                                                                                        ifelse(lab_bef$result_test_name == 'Neutrophil/Lymphocyte Ratio', 'Neutrophil Lymphocyte Ratio', 
#                                                                                                               ifelse(lab_bef$result_test_name == ' Monocytes Percentage', 'Monocytes',
#                                                                                                                      ifelse(lab_bef$result_test_name == ' Neutrophils', 'Neutrophils',
#                                                                                                                             ifelse(lab_bef$result_test_name == 'Neutrophils (Band)', 'Neutrophils (band)',
#                                                                                                                                    ifelse(lab_bef$result_test_name == '24Hr Phosphate', 'Phosphate 24Hr',
#                                                                                                                                           ifelse(lab_bef$result_test_name == 'Adjusted Calcium', 'Calcium (albumin adjusted)',
#                                                                                                                                                  ifelse(lab_bef$result_test_name == 'Alk. Phosphatase', 'Alkaline phosphatase',
#                                                                                                                                                         ifelse(lab_bef$result_test_name == 'C-Reactive Protein', 'CRP',
#                                                                                                                                                                ifelse(lab_bef$result_test_name == 'Fluid Glucose'|lab_bef$result_test_name == 'Glucose other', 'Glucose',
#                                                                                                                                                                       ifelse(lab_bef$result_test_name == 'Serum Folate', 'Folate - serum', 
#                                                                                                                                                                              ifelse(lab_bef$result_test_name == 'Hct', 'Haematocrit', 
#                                                                                                                                                                                     ifelse(lab_bef$result_test_name == 'MPV - Mean Platelet Volume', 'MPV', 
#                                                                                                                                                                                            ifelse(lab_bef$result_test_name == 'Reticulocytes Haemoglobin', 'RET-He',
#                                                                                                                                                                                                   ifelse(lab_bef$result_test_name == 'Serum B12', 'Vitamin B12 - serum',
#                                                                                                                                                                                                          ifelse(lab_bef$result_test_name == 'Uric Acid', 'Urate',lab_bef$result_test_name))))))))))))))))))))))))))

#Removing duplicates
#lab_bef = lab_bef %>% 
#  filter(!duplicated(cbind(id_patient, test_result_datetime, result_test_name, test_result)))

lab_bef = lab_bef %>% distinct(id_patient, test_result_datetime, result_test_name, test_result, .keep_all = TRUE)

#Calculating the timestamp
lab_bef = merge(lab_bef, nac, by = 'id_patient', all.x = TRUE)
lab_bef$Timestamp = as.numeric(difftime(lab_bef$test_result_datetime, 
                                        lab_bef$date_of_birth, units = 'days'))

#Do no run this --------------------------------------------------------------------------------
blood = function(test_name){
  test_name_1 = lab_bef %>% filter(result_test_name==test_name) %>% group_by(id_patient) %>% 
    mutate(test = paste("test", row_number(), sep = "_"))
  test_name_2 <- pivot_wider(test_name_1,id_cols = "id_patient", names_from = "test", 
                             values_from = "test_result")
  return(test_name_2)
}

blood_tests = list()
for (name in unique(lab_bef$result_test_name)) {
  blood_tests[[name]] = blood(name)
}

id_patient = soc10$id_patient
missing = setdiff(id_patient_seq, lab_bef$id_patient)

patient = data.frame(id_patient=missing)

#lab_bef = bind_rows(lab_bef, patient)
#-----------------------------------------------------------------------------------------------
### In long-format with timestamp

blood_long = function(test_name1){
  test_name_1 = lab_bef %>% filter(test_name==test_name1 & !is.na(test_result)) %>% group_by(id_patient)
  test_name_1 = test_name_1[,c(1,15,8)]
  test_name_1 = test_name_1 %>% filter(!is.na(Timestamp))
  #Change if it is necessary
  colnames(test_name_1) <- c('id_patient', paste('Timestamp',test_name1, sep = '_'), paste('test_result',test_name1, sep = '_'))
  return(test_name_1)
}

for (name in unique(lab_bef$test_name)) {
  print(name)
}

blood_tests_long = list()
for (name in unique(lab_bef$test_name)) {
  blood_tests_long[[name]] = blood_long(name)
}

standardize_test_result <- function(df) {
  df[,3] <- scale(df[,3])
  df[,2] <- scale(df[,2])
  return(df)
}

blood_tests_long_stand = lapply(blood_tests_long, standardize_test_result)
View(blood_tests_long_stand$Creatinine)

blood_tests_long_stand$Creatinine %>% ggplot() +
  geom_line(aes(Timestamp_Creatinine, test_result_Creatinine, group = id_patient,
                color = id_patient))

blood_tests_long$Creatinine %>% ggplot() +
  geom_line(aes(Timestamp_Creatinine, test_result_Creatinine, group = id_patient,
                color = id_patient))

for(i in 1:length(blood_tests_long_stand)){
  missing = setdiff(id_patient_seq, blood_tests_long_stand[[i]]$id_patient)
  patient = data.frame(id_patient=missing)
  blood_tests_long_stand[[i]] = bind_rows(blood_tests_long_stand[[i]], patient)
  blood_tests_long_stand[[i]][is.na(blood_tests_long_stand[[i]])]<-0
}

#To verify the data :)
View(blood_tests_long_stand$Creatinine)
#Flta = blood('Folate')

lab_bef %>% filter (test_name == 'Creatinine') %>% ggplot() +
  geom_line(aes(test_result_datetime, test_result, group = id_patient,
                color = id_patient))

#Extracting a static variable

lab_bef1 = lab_bef %>% group_by(id_patient) %>% summarise('number_tests'=n()) 
lab_bef2 = lab_bef %>% filter(test_result_abnormal!='N'&test_result_abnormal!='y') %>%
  group_by(id_patient) %>% summarise('abnormal_test'=n())
lab_bef3 = merge(lab_bef1, lab_bef2, by = 'id_patient', all.x = T)
lab_bef3[is.na(lab_bef3)]<-0
#lab_bef3$abnormal_rate = (lab_bef3$abnormal_test/lab_bef3$number_tests)*100

soc6 = merge(soc5, lab_bef3, by = 'id_patient', all.x = TRUE)
#soc7 = merge(soc6, Dem,  by = 'id_patient', all.x = TRUE)

#4g. Calculate P3 index 

#source("p3index_scoring.R")
#medicine = read_xlsx('List_Medicine_P3_v3.xlsx')
#medicine$Chemical = tolower(medicine$Chemical)
#Med_bef$`Pharmacy List` = tolower(Med_bef$`Pharmacy List`)
#Med_bef1 = merge(Med_bef, medicine, by.x = 'Pharmacy List', by.y = 'Chemical', all.x = T)
#p3 = p3index_scoring(Med_bef1, therapeutic_group_code = 'PHARMAC.codes', 
#                     return_condition_cols = FALSE, id_cols = 'id_patient')

drug = read_xlsx("names_medicine_final - TG.xlsx", sheet = 'names_medicine_final',
                 trim_ws = FALSE)

drug$TG2_C = ifelse(grepl('Diabetes', drug$TG2, ignore.case = TRUE), 'Diabetes',
                    ifelse(drug$TG1 == "Special Foods", "Special Foods", drug$TG2))

drug$TG2_S = ifelse(grepl('Diabetes', drug$TG2, ignore.case = TRUE), 'Diabetes',
                    ifelse(drug$TG1 == "Cardiovascular System", "Cardiovascular", 
                           ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Analgesics" & (drug$TG3 == "Opioid Analgesics"|drug$TG3 == "OPIOIDS"), "Analgesics",
                                  ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Agents for Parkinsonism and Related Disorders", "Agents for Parkinsonism and Related Disorders",
                                         ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Antidepressants", "Antidepressants",
                                                ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Antipsychotic Agents", "Antipsychotics",
                                                       ifelse(drug$TG1 == "Nervous System" & drug$TG2 == "Anxiolytics", "Anxiolytics", NA)))))))

Med_bef = merge(Med_bef, drug, by = 'generic_name', all.x = TRUE)
Med_bef = Med_bef %>% filter(!is.na(TG2_C))


Med_bef1 = Med_bef %>% group_by(id_patient, TG2_C) %>% 
  summarise('num_dispense' = n())

Med_bef1 <- pivot_wider(Med_bef1, id_cols = "id_patient", names_from = "TG2_C", 
                        values_from = "num_dispense")
Med_bef1[is.na(Med_bef1)]<-0

soc8 = merge(soc6, Med_bef1,  by = 'id_patient', all.x = TRUE)

#4h. from ARC, extract the total days per service category. 

arc_bef1 = arc_bef %>% group_by(id_patient, `Service.Category`) %>% 
  summarise('Tot_days'=sum(`No..of.Units`, na.rm = T))
arc_bef1 <- pivot_wider(arc_bef1, id_cols = "id_patient", names_from = "Service.Category", 
                        values_from = "Tot_days")
arc_bef1[is.na(arc_bef1)]<-0

soc9 = merge(soc8, arc_bef1,  by = 'id_patient', all.x = TRUE)
