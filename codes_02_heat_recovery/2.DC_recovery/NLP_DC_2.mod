reset;
################################
# DTmin optimisation
################################
# Sets & Parameters

set Time default {};          #your time set from the MILP part
param Qheating{Time};         #[kWh] your heat demand from the MILP part, only Medium temperature heat (65 deg C)
param top{Time};              #[h] your operating time from the MILP part

param EPFLMediumT 	:= 65;    #[degC] - desired temperature high temperature loop
param EPFLMediumOut := 30;    #[degC] - temperature of return low temperature loop

param TDCin 		:= 60;    #[degC] temperature of air coming from data center into the heat recovery HE
param UDC 			:= 0.15;  #[kW/(m2 K)] air-water heat transfer coefficient

param CarnotEff 	:= 0.55;  #[-] assumption: carnot efficiency of heating heat pumps
param Cel 			:= 0.20;  #[CHF/kWh] operating cost for buying electricity from the grid

param THPhighin 	:= 7;     #[degC] temperature of water coming from lake into the evaporator of the HP
param THPhighout 	:= 3;     #[degC] temperature of water coming from lake into the evaporator of the HP

param i 			:= 0.06; #interest rate
param n 			:= 20;    #[y] life-time
param FBMHE 		:= 4.74;  #[-] bare module factor of the heat exchanger
param INew 			:= 605.7; #[-] chemical engineering plant cost index (2015)
param IRef 			:= 394.1; #[-] chemical engineering plant cost index (2000)
param aHE 			:= 800;   #[-] HE cost parameter
param bHE 			:= 0.6;   #[-] HE cost parameter

param HeatDC 		:= 574;  #[kW] amount of heat to be removed from data center
param Tret 			:= 17;   #[degC] temperature of air entering DC
param MassDC 		:= HeatDC/(TDCin-Tret); #[KJ/(s degC)] MCp of air in DC
param Cpwater		:= 4.18; #[kJ/(kg degC)]
################################
##Variables

var TDCout{Time} 	>= 0.001; #[degC] temperature of air coming from data center out of the heat recovery HE
var AHEDC 			>= 0.001; #[m2] area of heat recovery heat exchanger
var dTLMDC{Time} 	>= 0.001; #[K] logarithmic mean temperature difference in the heat recovery heat exchanger
var TRadin{Time} 	>= 30.01; #[degC]

var E{Time} 		>= 0.001; #[kW] electricity consumed by the heat pump (using pre-heated lake water)
var TLMCond{Time} 	>= 0.001; #[K] logarithmic mean temperature in the condensor of the heating HP (using pre-heated lake water)
var Qevap{Time} 	>= 0.001; #[kW] heat extracted in the evaporator of the heating HP (using pre-heated lake water)
var Qcond{Time} 	>= 0.001; #[kW] heat delivered in the condensor of the heating HP (using pre-heated lake water)
var COP{Time} 		>= 0.001; #[-] coefficient of performance of the heating HP (using pre-heated lake water)

var OPEX 			>= 0.001; #[CHF/year] operating cost
var CAPEX 			>= 0.001; #[CHF/year] annualized investment cost
var TC 				>= 0.001; #[CHF/year] total cost

var TLMEvapHP{Time} >= 0.001; #[K] logarithmic mean temperature in the evaporator of the heating HP

var Qrad{Time} 		>= 0.001; #[kW] DC heat recovered;

var THPin{Time} 	>= 7;     #[degC] ?
var Qfree{Time} 	>= 0.001; #[kW] free cooling heat; makes sure DC air is cooled down.
var Flow{Time} 		>= 0.001; #[kg/s] lake water entering free cooling HEX
var MassEPFL{Time} 	>= 0.001; #[kJ/(s degC)] MCp of EPFL heating system 

################################
# Constraints
####### Direct Heat Exchanger;

## TEMPERATURE CONTROL CONSTRAINS exist to be sure the temperatures in the HEX do not cross, meaning to make sure there is a certain DTmin. (3 are recommended, but you can have more or less)
subject to Tcontrol1{t in Time}: 
    TDCin>=TRadin[t]+0.1; #[degC]=[degC]

subject to Tcontrol2 {t in Time}:
    TDCout[t]>=EPFLMediumOut+0.1; #[degC]=[degC]

subject to Tcontrol3 {t in Time}:
 	TDCin>=TDCout[t]+0.1; #[degC]=[degC]

subject to Tcontrol4 {t in Time}:
 	TRadin[t]>=EPFLMediumOut+0.1; #[degC]=[degC]

subject to Tcontrol5 {t in Time}: 
    TDCout[t] >= THPin[t]+5;

## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION
subject to dTLMDataCenter {t in Time}: #the logarithmic mean temperature difference in the heat recovery HE can be computed
    dTLMDC[t]*log((TDCin-TRadin[t])/(TDCout[t]-EPFLMediumOut))=((TDCin-TRadin[t])-(TDCout[t]-EPFLMediumOut)); #[K]*[-] = [K]

subject to HeatBalance1{t in Time}: #Heat balance in DC HEX from DC side
    Qrad[t]=MassDC*(TDCin-TDCout[t]); #[kW] = [kJ/(s degC)]*[degC]

# subject to Flows1:
	# MassDC = HeatDC/(TDCin-Tret);

# subject to Flows2{t in Time}:
	# MassEPFL[t] = (Qheating[t]/top[t])/(EPFLMediumT-EPFLMediumOut);	

# subject to HeatBalance2{t in Time}:
	# MassDC*(TRadin[t]-EPFLMediumOut) = Qrad[t];	

subject to AreaHEDC{t in Time}: #the area of the heat recovery HE can be computed using the heat extracted from DC, the heat transfer coefficient and the logarithmic mean temperature difference 
    AHEDC>=Qrad[t]/(UDC*(dTLMDC[t])); #[m2]=[kW]/[kW/(m2 K)]/[K]

subject to balancemax{t in Time}: # the maximum heat extracted is for sure lower than the total heating demand; pay attention to the units!
	Qrad[t] = MassEPFL[t]*(TRadin[t]-EPFLMediumOut); #[kW]=[kJ/(s degC)]*[degC]
	
## MEETING HEATING DEMAND, ELECTRICAL CONSUMPTION

subject to Freecooling1{t in Time}: #Free cooling from one side
    Qfree[t]=MassDC*(TDCout[t]-Tret); #[kW]=[kJ/(s degC)]*[degC]

subject to Freecooling2{t in Time}: #Free cooling from the other side
    Qfree[t]=Flow[t]*Cpwater*(THPin[t]-THPhighin); 

subject to QEvaporator{t in Time}: #water side of evaporator that takes flow from Free cooling HEX
    Qevap[t]=Flow[t]*Cpwater*(THPin[t]-THPhighout); 

subject to QCondensator{t in Time}: #EPFL side of condenser delivering heat to EFPL
	Qcond[t]=MassEPFL[t]*(EPFLMediumT-TRadin[t]);
	
subject to HeatBalanceDC{t in Time}: #makes sure all HeatDC is removed;
    Qrad[t]+Qfree[t]=HeatDC;

subject to Electricity1{t in Time}: #the electricity consumed in the HP can be computed using the heat delivered and the heat extracted
    E[t]+Qevap[t]=Qcond[t];

subject to Electricity{t in Time}: #the electricity consumed in the HP can be computed using the heat delivered and the COP
    E[t]*COP[t]=Qcond[t];

subject to COPerformance{t in Time}: #the COP can be computed using the carnot efficiency and the logarithmic mean temperatures in the condensor and in the evaporator
    COP[t]*(TLMCond[t]-TLMEvapHP[t])=CarnotEff*(TLMCond[t]);

subject to dTLMCondensor{t in Time}: #the logarithmic mean temperature on the condenser, using inlet and outlet temperatures. Note: should be in K
    #TLMCond[t]*log((EPFLMediumT+273)/(TRadin[t]+273))=(EPFLMediumT-TRadin[t]);
    TLMCond[t]=(EPFLMediumT + TRadin[t])/2 + 273; #[K] linearized approximation

subject to dTLMEvaporator{t in Time}: #the logarithmic mean temperature can be computed using the inlet and outlet temperatures, Note: should be in K
    #TLMEvapHP[t]*log((THPin[t]+273)/(THPhighout+273))=(THPin[t]-THPhighout); 
    TLMEvapHP[t]=(THPin[t] + THPhighout)/2 + 273; #[K] linearized approximation

subject to QEPFLausanne{t in Time}: #the heat demand of EPFL should be the sum of the heat delivered by the 2 systems;
    Qheating[t]/top[t]=Qcond[t]+Qrad[t];

## COSTS and OBJECTIVE
subject to OPEXcost: #the operating cost can be computed using the electricity consumed in the HP
    OPEX=sum{t in Time} (Cel*top[t]*E[t]);

subject to CAPEXcost: #the investment cost can be computed using the area of the heat recovery heat exchanger and annuity factor
    CAPEX=(i*(1+i)^n)/((1+i)^n-1)*FBMHE*(INew/IRef)*aHE*AHEDC^bHE;

subject to TCost: #the total cost can be computed using the operating and investment cost
    TC=OPEX+CAPEX;

################################
minimize obj : TC;
