/*---------------------------------------------------------------------------------------------------------------------------------------
Set the efficiency of PV
---------------------------------------------------------------------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------------------------------------------------------------------
Set the electricity output as a function of irradiation and efficiency 
---------------------------------------------------------------------------------------------------------------------------------------*/
var usedroofAreaPV >= 0;
#var usedroofAreaSTC >= 0;

param eff_PV := 0.15;
#param eff_STC := 0.836;
#param Tlm := 61.22;


subject to pv_elec{t in Time}:
	Flowout['Electricity','PV'] * mult_t['PV',t] = irradiation[t] * usedroofAreaPV * eff_PV;

#subject to STC_heat{t in Time}:
#	irradiation[t]!=0 ==> Qheatingsupply['STC'] * mult_t['STC',t] = usedroofAreaSTC * (irradiation[t] * eff_STC - 4.16/1000*(Tlm-Text[t]) - 0.073/1000*(Tlm-Text[t])^2) else Qheatingsupply['STC'] * mult_t['STC',t] = 0;
/*---------------------------------------------------------------------------------------------------------------------------------------
roof usage 
---------------------------------------------------------------------------------------------------------------------------------------*/
subject to roof_cstr:
	roofArea >= usedroofAreaPV;# + usedroofAreaSTC;


#let cinv1["STC"]:=0.087*347;#*usedroofAreaSTC; # CHF/year sr: Stadler P.,  Model-based sizing of building energy systems with renewable sources
#let cinv2["STC"]:=0.087*126*usedroofAreaSTC/mult['STC']; # CHF/year (1000W/m2 and 1 m2) sr: Stadler P., Model-based sizing of building energy systems with renewable sources
#let cinv2["STC"]:=0.087*126*3.61;
