from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

modelo = BayesianNetwork([("Fuel", "Start"), ("TurnOver", "Start"), ("Battery", "TurnOver"), ("Battery", "Gauge"), ("Fuel", "Gauge")])

Battery = TabularCPD('Battery', 2, [[0.02], [0.98]])
Fuel = TabularCPD('Fuel', 2, [[0.05], [0.95]])
Gauge = TabularCPD('Gauge', 2, [[0.99, 0.97, 0.1, 0.04], [0.01, 0.03, 0.9, 0.96]], evidence=['Battery', 'Fuel'], evidence_card=[2, 2])
TurnOver = TabularCPD('TurnOver', 2, [[0.03, 0.98], [0.97, 0.02]], evidence=['Battery'], evidence_card=[2])
Start = TabularCPD('Start', 2, [[0.01, 1.0, 0.92, 0.99], [0.99, 0.0, 0.08, 0.01]], evidence=['TurnOver', 'Fuel'], evidence_card=[2, 2])

modelo.add_cpds(Battery, Fuel, Gauge, TurnOver, Start)

modelo.check_model()

infer = VariableElimination(modelo)
posterior_p = infer.query(["Fuel"], evidence={"Start": 0})
print(posterior_p)

infer = VariableElimination(modelo)
posterior_p = infer.query(["Battery"], evidence={"Start": 0})
print(posterior_p)