import gym
from gym import error, spaces, utils
from gym.utils import seeding
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import RWMol
from rdkit import RDConfig
import os
fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

class MoleculeEnvironment(gym.Env):
    """
    Description:
        The drug begins in one of two initial conditions a single carbon atom 
        or a random starting molecule. At each step there is an addition or 
        removal of functional groups.
    Source:
        
    Observation:
        Type: 
        Num     Observation     Value         
        
    Actions:
        Type: 
        Num     Action
       
    Reward:
        
    Starting State:
        
    Episode Termination:
        
    """
    
    def __init__(self):
        self.current_molecule  = RWMol(Chem.MolFromSmiles('C'))  
        self.obs = Observation(self.current_molecule)
        self.molecule_list = ['C']
        self.mol_Steps =[]
        self.smiles = []
        
    def step(self,action_ob ):
        if action_ob.action_c == "Add":
            if action_ob.pos == 'front':
                self.molecule_list.insert(0,action_ob.mol)
            elif action_ob.pos == 'back':  
                self.molecule_list.append(action_ob.mol)
        elif action_ob.action_c == "Remove":
            if action_ob.pos == 'front':
                self.molecule_list.remove(self.molecule_list[0])
            elif action_ob.pos == 'back':  
                self.molecule_list.pop()
                     
        self.current_molecule = RWMol(Chem.MolFromSmiles( self.listToSmiles()))  
        
        self.obs.update(self.current_molecule) 
        self.mol_Steps.append(self.current_molecule)
        legend = str(len(self.mol_Steps))+ ". " + Chem.MolToSmiles(self.current_molecule)
        self.smiles.append(legend) 
        return self.obs    

    def reset(self):
        self.current_molecule  = RWMol(Chem.MolFromSmiles('C'))

    def render(self):
        if len(self.mol_Steps) < 4:
            img = Draw.MolsToGridImage(self.mol_Steps, molsPerRow = len(self.mol_Steps), legends = [str(x) for x in self.smiles])
        else:
            img = Draw.MolsToGridImage(self.mol_Steps, molsPerRow = 4, legends = [str(x) for x in self.smiles])
        
        return img


    def seed(self,Smiles):
        self.current_molecule  = RWMol(Chem.MolFromSmiles(Smiles))  
        self.molecule_list = [Smiles]
        
    def listToSmiles(self):
        smiles = ''
        for mol_str in self.molecule_list:
            smiles += mol_str
        return smiles
        
class Action():
    def __init__(self):
        self.action_c = ''
        self.pos = ''   #front or back
        self.mol = ''
             
    def setAction(self,action,pos,mol='C'): #mol
        self.action_c = action
        self.mol = mol
        self.pos = pos 
        
class Observation:
    
    def __init__(self, mol):
        self.mol = mol
        self.observation = Chem.MolToSmiles(mol)
        self.info = []
    
    def getInfo(self):
        self.info.clear()
        feats = factory.GetFeaturesForMol(self.mol)
         
        for y in feats:
            self.info.append(y.GetType())
                
        return self.info
    
    def getObservation(self):
        self.observation = Chem.MolToSmiles(self.mol)
        return self.observation
        
    def update(self,mol):
        self.mol = mol         