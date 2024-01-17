################################
# DTmin optimisation
################################
# Sets & Parameters
reset;
set Time default {};        				#your time set from the MILP 
set Buildings default {};					# set of buildings
for {n in {1..24}}{let Buildings:= Buildings union {'Building'&n};}

set MediumTempBuildings default {};			# set of buildings heated by medium temperature loop
set med_T_id = {1, 2, 3, 4, 5 ,6 ,7, 8, 9, 10};
for {n in {med_T_id}}{let MediumTempBuildings:= MediumTempBuildings union {'Building'&n};}

set LowTempBuildings default {};			# set of buildings heated by low temperature loop
set low_T_id = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
for {n in {low_T_id}}{let LowTempBuildings:= LowTempBuildings union {'Building'&n};}


param Text{t in Time};  #external temperature - Part 1
param top{Time}; 		#your operating time from the MILP part
#param Areabuilding		>=0.001; #defined .dat file.

param Tint 				:= 21; # internal set point temperature [C]
param mair 				:= 2.5; # m3/m2/h
param Cpair 			:= 1.152; # kJ/m3K
param Uvent 			:= 0.025; # air-air HEX
#param Uair_ref 			:= 0.15; # air-air HEX
#param ksun 				:= 0.05; # ksun value


param EPFLMediumT 		:= 65; #[degC]
param EPFLMediumOut 	:= 30; 

param CarnotEff 		:= 0.55; #assumption: carnot efficiency of heating heat pumps
param Cel 				:= 0.20; #[CHF/kWh] operating cost for buying electricity from the grid

param THPhighin 		:= 7; #[deg C] temperature of water coming from lake into the evaporator of the HP
param THPhighout 		:= 3; #[deg C] temperature of water coming from lake into the evaporator of the HP

param i 				:= 0.06 ; #interest rate
param n 				:= 20; #[y] life-time
param FBMHE 			:= 4.74; #bare module factor of the heat exchanger
param INew 				:= 605.7; #chemical engineering plant cost index (2015)
param IRef 				:= 394.1; #chemical engineering plant cost index (2000)
param aHE 				:= 1200; #HE cost parameter
param bHE 				:= 0.6; #HE cost parameter

################################
# Variables
var Text_new{Time}  ; # air Temperature after air-air HEX;
var Trelease{Time}  ; #[degC]
var Qheating{Time} 	>= 0.001; #your heat demand from the MILP part, will become a variable in the case of heat recovery from air ventilation

var E{Time} 		>= 0.001; # [kW] electricity consumed by the reference heat pump (using pre-heated lake water)
var TLMCond 	 	>= 0.001; #[K] logarithmic mean temperature in the condensor of the heating HP (using pre-heated lake water)
var TLMEvap 		>= 0.001; # K] logarithmic mean temperature in the evaporator of the heating HP (using pre-heated lake water)
var Qevap{Time} 	>= 0.001; #[kW] heat extracted in the evaporator of the heating HP (using pre-heated lake water)
var Qcond{Time} 	>= 0.001; #[kW] heat delivered in the condensor of the heating HP (using pre-heated lake water)
var COP{Time} 		>= 0.001; #coefficient of performance of the heating HP (using pre-heated lake water)

var OPEX 			>= 0.001; #[CHF/year] operating cost
var CAPEX 			>= 0.001; #[CHF/year] annualized investment cost
var TC 				>= 0.001; #[CHF/year] total cost

var TLMEvapHP 		>= 0.001; #[K] logarithmic mean temperature in the evaporator of the heating HP (not using pre-heated lake water

var Heat_Vent{Time} >= 0; 	  #[kW]
var DTLNVent{Time} 	>= 0.001; #[degC]
var Area_Vent 		>= 0.001; #[m2]
var DTminVent 		>= 2;     #[degC]

var Flow{Time} 		>= 0.001; #lake water entering free coling HEX
var MassEPFL{Time} 	>= 0.001; # MCp of EPFL heating system [KJ/(s degC)]

var Uenv{Buildings} >= 0; # overall heat transfer coefficient of the building envelope 

## Variables and parameters Air-Air HP

param Cref_hp				:= 3400;
param beta_hp				:= 0.85;
param BM_hp					:= 2;
param MS2000				:= 400;
param MS2017				:= 562;

var Trelease_2{Time}     	>=0.001; #release temperature (check drawing);    
var Tair_in{Time}        	<= 40;   #lets assume EPFL cannot take ventilation above 40 degrees (safety)
var Cost_HP       		 	>=0.001; #HP cost 

var E_2{Time} 				>= 0.00001; # [kW] Electricity used in the Air-Air HP
var TLMCond_2{Time} 		>= 273; 	#[K] logarithmic mean temperature in the condensor of the new HP 
var TLMEvapHP_2{Time} 		>= 200; 	# K] logarithmic mean temperature in the evaporator of the new HP 
var Qevap_2{Time} 			>= 0.0001;  #[kW] heat extracted in the evaporator of the new HP 
var Qcond_2{Time} 			>= 0.0001;  #[kW] heat delivered in the condensor of the new HP 
var COP_2{Time} 			>= 0.0001;  #coefficient of performance of the new HP 

#### new code

param irradiation{Time};				 # solar irradiation [kW/m2] at each time step									
param specElec{Buildings} default 0.04;  #electricity consumption of buildings [kW/m2]{Buildings}
param FloorArea{Buildings} default 1000; #area [m2]{Buildings}
param k_th{Buildings} default 0.006;	 # thermal losses and ventilation coefficient in (kW/m2/K)
param k_sun{Buildings} default 0.10;	 # solar radiation coefficient [−]{Buildings}
param share_q_e default 0.8; 			 # share of internal gains from electricity [-]{Buildings}
param specQ_people{Buildings} default 0.00544; # specific average internal gains from people [kW/m2]{Buildings}

param eps := 0.001;

################################
# Constraints
################################
## VENTILATION

subject to la_contrainte_a_la_con:
	Area_Vent<=1500;

subject to overallHeatTransfer{b in MediumTempBuildings}: # Uenv calculation for each building based on k_th and mass of air used
	Uenv[b] = k_th[b] - mair*Cpair/3600;

subject to VariableHeatdemand {t in Time}: #Heat demand calculated as the sum of all buildings -> medium temperature
	Qheating[t] = sum{b in MediumTempBuildings} max(FloorArea[b]*(Uenv[b]*(Tint-Text[t]+eps) + mair*Cpair/3600*(Tint-Tair_in[t]+eps) - k_sun[b]*irradiation[t]-specQ_people[b]- share_q_e*specElec[b]),0);

subject to Heat_Vent1 {t in Time}: #HEX heat load from one side;
	Heat_Vent[t] = sum{b in MediumTempBuildings} (Text_new[t]-Text[t])*(mair/3600)*Cpair*FloorArea[b];

subject to Heat_Vent2 {t in Time}: #HEX heat load from the other side;
	Heat_Vent[t] = sum{b in MediumTempBuildings} (Tint-Trelease[t])*(mair/3600)*Cpair*FloorArea[b]; #CHANGED

# subject to DTLNVent1 {t in Time}: #DTLN ventilation
# 	DTLNVent[t] * log((Tint-Text_new[t]+eps)/(Trelease[t]-Text[t]+eps)) = ((Tint-Text_new[t]+eps)-(Trelease[t]-Text[t]+eps)) ; #CHANGED
subject to DTLNVent1 {t in Time}:
	DTLNVent[t]=((Tint-Text_new[t])+(Trelease[t]-Text[t]))/2;

subject to Area_Vent1 {t in Time}: #Area of ventilation HEX
	Area_Vent >= Heat_Vent[t]/(Uvent*DTLNVent[t]);

subject to DTminVent1 {t in Time}: #DTmin needed on one side of HEX
	Trelease[t] >= DTminVent+Text[t];

subject to DTminVent2 {t in Time}: #DTmin needed on the other side of HEX 
	Tint >= DTminVent+Text_new[t];

subject to ventilation_trivial {t in Time}: #relation between Text_new and Text (initialization purposes)
	Text_new[t] >= Text[t]+eps;

################################
# Constraints
################################

## MASS BALANCE

subject to Flows{t in Time}: #MCp of EPFL heating fluid calculation.
	MassEPFL[t]*(EPFLMediumT-EPFLMediumOut) >= (Qheating[t]);	

## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

subject to QEvaporator{t in Time}: #water side of evaporator that takes flow from Free cooling HEX
	Qevap[t] = Flow[t]*4.18*(THPhighin - THPhighout);

subject to QCondensator{t in Time}: #water side of evaporator that takes flow from Free cooling HEX
	Qcond[t] = MassEPFL[t]*(EPFLMediumT - EPFLMediumOut);	

subject to Electricity{t in Time}: #the electricity consumed in the HP can be computed using the heat delivered and the heat extracted (Reference case)
	E[t] = Qcond[t] - Qevap[t];

subject to Electricity_1{t in Time}: #the electricity consumed in the HP can be computed using the heat delivered and the COP (Reference case)
	E[t]*COP[t] = Qcond[t];

subject to COPerformance{t in Time}: #the COP can be computed using the carnot efficiency and the logarithmic mean temperatures in the condensor and in the evaporator
	COP[t] = CarnotEff* TLMCond/(TLMCond - TLMEvapHP);

subject to dTLMCondensor{t in Time}: #the logarithmic mean temperature on the condenser, using inlet and outlet temperatures. Note: should be in K (Reference case)
	TLMCond =(EPFLMediumOut-EPFLMediumT)/log((EPFLMediumOut+273)/(EPFLMediumT+273));

subject to dTLMEvaporatorHP{t in Time}: #the logarithmic mean temperature can be computed using the inlet and outlet temperatures, Note: should be in K (Reference case)
	TLMEvapHP = (THPhighin - THPhighout)/log((THPhighin+273.15)/(THPhighout+273.15));

## Air Air HP

subject to temperature_gap{t in Time}: #relation between Text and Text_new;
	Text_new[t] >= Text[t]+eps;

subject to temperature_gap2{t in Time}: #relation between Trelease and Trelease2;
	Trelease[t] >= Trelease_2[t]+eps;

subject to temperature_gap3{t in Time}: # relation between Tair_in and Text_new;
	Tair_in[t]  >= Text_new[t]+eps;

 subject to QEvaporator_2{t in Time}: #Evaporator heat from air side
	Qevap_2[t] = 0.8*sum{b in MediumTempBuildings} FloorArea[b]*(mair/3600)*Cpair*(Trelease[t] - Trelease_2[t]);

subject to QCondensator_2{t in Time}: #Condeser heat from air side
	Qcond_2[t] = sum{b in MediumTempBuildings} FloorArea[b]*(mair/3600)*Cpair*(Tair_in[t] - Text_new[t]);	

subject to Electricity_2{t in Time}: #the electricity consumed in the new HP can be computed using the heat delivered and the heat extracted
	E_2[t] = Qcond_2[t] - Qevap_2[t];

subject to Electricity_3{t in Time}: #the electricity consumed in the new HP can be computed using the heat delivered and the COP
	E_2[t]*COP_2[t] = Qcond_2[t];

subject to COPerformance_extra{t in Time}: #COP cannot exceed 7;
	COP_2[t] <= 7;

subject to COPerformance_extra_2{t in Time}: #Assume that COP has to be higher than 3
	COP_2[t] >= 3;		

subject to COPerformance_2{t in Time}: #the COP can be computed using the carnot efficiency and the logarithmic mean temperatures in the condensor and in the evaporator
	COP_2[t]*(TLMCond_2[t]-TLMEvapHP_2[t]) = CarnotEff*TLMCond_2[t];

subject to dTLMCondensor_2{t in Time}: #the logarithmic mean temperature in the new condenser. Note: should be in K
	TLMCond_2[t]*log((Tair_in[t]+273)/(Text_new[t]+273))=(Tair_in[t]-Text_new[t]);

subject to dTLMEvaporatorHP_2{t in Time}: #the logarithmic mean temperature in the new Evaporator, Note: should be in K
	TLMEvapHP_2[t]*log((Trelease[t]+273.15)/(Trelease_2[t]+273.15)) = (Trelease[t] - Trelease_2[t]);

## COST CONSIDERATIONS

subject to Costs_HP {t in Time}: # new HP cost
	Cost_HP >= Cref_hp*((E_2[t])**(beta_hp))*(MS2017/MS2000)*BM_hp;

subject to QEPFLausanne{t in Time}: #the heat demand of EPFL should be met;
	Qheating[t] = Qcond[t];

subject to OPEXcost: #the operating cost can be computed using the electricity consumed in the two heat pumps
	OPEX = sum{t in Time}(Cel*(E[t]+E_2[t])*top[t]);

subject to CAPEXcost: #the investment cost can be computed using the area of ventilation HEX and new HP.
	CAPEX = (i*(1+i)^n)/((1+i)^n - 1)*(Cost_HP+FBMHE*(INew/IRef)*aHE*Area_Vent^bHE);

subject to TCost: #the total cost can be computed using the operating and investment cost
	TC = OPEX + CAPEX;

################################
minimize obj : TC;