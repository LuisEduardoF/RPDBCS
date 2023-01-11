# Installation
```
git clone http://gitlab.ninfa.inf.ufes.br/lucashsmello/active-learning
cd active-learning
pip install -r requirements.txt
pip install git+http://gitlab.ninfa.inf.ufes.br/ninfa-ufes/deep-rpdbcs#subdirectory=src/python
```

# Usage
```
cd rpdbcs/active_learning
python validation.py -i data/data_classified_v6 -o results.csv -c ../../myconfig.yaml
```
O arquivo myconfig.yaml contém as configurações experimentais desejadas.