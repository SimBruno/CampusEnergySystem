/*---------------------------------------------------------------------------------------------------------------------------------------
Set the efficiency of HP2
---------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------
Set the electricity input 
---------------------------------------------------------------------------------------------------------------------------------------*/
param a:= 1;
param b:= 1;
param c:= 1;

param EPFLMediumT 	:= 338; #[degK] - desired temperature high temperature loop
param EPFLMediumOut := 303; # temperature of return low temperature loop [degK]
param THPhighin 	:= 280; #[deg K] temperature of water coming from lake into the evaporator of the HP
param THPhighout 	:= 276; #[deg K] temperature of water coming from lake into the evaporator of the HP

param TLMCond := (EPFLMediumOut-EPFLMediumT)/(log(EPFLMediumOut/EPFLMediumT)); #Assume cste

param TLMEvapHP := (THPhighin-THPhighout)/(log((THPhighin)/(THPhighout))); 

subject to HP2_elec{t in Time}:
	Flowin['Electricity','HP2']* mult_t['HP2',t] = Qheatingsupply['HP2']*(TLMCond-TLMEvapHP)/(TLMCond*(a*Text[t]^2-b*Text[t]+c));


