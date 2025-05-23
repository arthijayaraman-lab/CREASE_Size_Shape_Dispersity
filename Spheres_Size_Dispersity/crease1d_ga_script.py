# CREASE genetic algorithm for analyzing 1D scattering profiles for spherical nanoparticles with dispersity in particle size.
# Code written by Prof. Arthi Jayaraman's research group at the University of Delaware, Newark, Delaware. (2024).  

#import all necessary files
import numpy as np
import sys
import xgboost as xgb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')
    
# 3 structural features for this system are:
# mean and standard deviation of volumetric radii.
# mean and standard deviation of aspect ratio.
# volume fraction.

# Defining the ranges for 3 structural features.

# Range for mean volumetric radii of the system is 5-500 Angstroms
# Rmu divided into 3 ranges of 5-20 Angstroms, 20-100 Angstroms, 100-500 Angstroms.
rmu_min = 5.0
rmu_max = 500.0

# Range for standard deviation of the volumetric radii is 0-100% of the mean volumetric radii
rsig_min = 0.0
rsig_max = 1.0

# Range for volume fraction in this work is 0.05 - 0.15
phi_min = 0.05
phi_max = 0.15

# When comparing experimental profiles to computed profiles, we need a factor (f) that scales the computed profile and a-
# - constant parameter (c) that accounts for uncertainities in the background subtraction.
# From our manual matching stage, we also identify a range for the factor (f), and (c).
# The factor (f) and constant parameter (c) are also chosen by the genetic algorithm.

f_min =  0.008
f_max = 0.063

c_min = 1.7
c_max = 3.4

# Function that calculates the fitness of GA Individuals.
# For matching 1D computed profiles to experimental data we use the log-based error function from the reference:
# Zijie Wu and Arthi Jayaraman, Macromolecules (2022), 55.24, 11076-11091

mse_cut = 5*10**-5 # Out of the 100 GA Individuals MSE cut off only choses the individual with better fitness (lower MSE) than this value. If you wish to include all 100 individuals set this to a random high value like 100.0

def calculate_fitness(data1, data2, f, c):

    data2                                = 10**data2
    data2                                = data2*f + c

    log_error = 0.0	
    
    for i in range(numq):
        log_error = log_error + weights[i]*(np.log(data1[i]/data2[i])**2.0)  
    	
    log_error = log_error/numq

    return log_error

# Function to create a matrix of structural features, and q to be input to the XGBoost ML model
def create_combined_data(struc_features):
    shape_struc_features=struc_features.shape
    shape_q = q.shape
    repeated_struc_features = np.repeat(struc_features, repeats=shape_q, axis=0)
    repeated_q = (np.tile(q, shape_struc_features[0])).reshape(-1, 1)
    combined_data = np.hstack((repeated_struc_features,repeated_q))
    return combined_data

# Function to translate genes to structural features for all generations.  
def genes_to_struc_features(genevalues): 
    phi = genevalues[:,2]*(phi_max - phi_min) + phi_min
    shape_genes = genevalues.shape[0]
    meanR = np.zeros(shape_genes)   
 
#  Rmu is divided into 3 ranges 5-20, 20-100, 100-500.
# The 20-100 Angstrom range is given a higher gene range because it is the most probable
# Range 1 (5-20) Angstroms which is represented by gene values between 0-0.1
# Range 2 (20-100) Angstroms which is represented by gene values between 0.1-0.9
# Range 3 (100-500) Angstroms which is represented by gene values between 0.9-1.0
    for j in range(shape_genes):
        if (genevalues[j, 0] <= 0.1):
            meanR[j] = genevalues[j,0]*150 + 5.0
        elif (genevalues[j, 0] <= 0.9):
            meanR[j] = (genevalues[j, 0] - 0.1)*100.0 + 20.0
        else:
            meanR[j] = (genevalues[j, 0] - 0.9)*4000.0 + 100.0

    sigmaR = genevalues[:,1]*(rsig_max - rsig_min) + rsig_min
    struc_features = np.vstack((meanR, sigmaR, phi))
    struc_features=struc_features.transpose()
    return struc_features

# Function to compute scattering profile from structural features using the XGBoost model.
def generate_profile(combined_data):
    feature_names = ["Mean R", "Std R", "true_vol_frac", "q"]
    dmatrix = xgb.DMatrix(combined_data, feature_names=feature_names)
    generated_profile = xgboost_model.predict(dmatrix)
    return generated_profile


#    Generate profiles for all individuals in the genetic table .
#
#    Parameters:
#        - gatable (np.ndarray): Genetic table containing gene values.
#        - profile_generator (ProfileGenerator): An instance of the ProfileGenerator class.
#
#    Returns:
#        - np.ndarray: Generated profiles for all individuals.

def generateallprofiles(gatable):

    popsize=gatable.shape[0]
    indscore=gatable.shape[1]-1
    strucfeatures = genes_to_struc_features(gatable[:,0:indscore])
    combined_data = create_combined_data(strucfeatures)
    shape_combined_data=combined_data.shape
    generated_profiles=np.zeros((numq,popsize))
    for n in range(popsize):
        inputdata=combined_data[int(n*numq):int((n+1)*numq),:]
        generated_profile=generate_profile(inputdata)
        generated_profiles[:,n]=generated_profile
    return generated_profiles


#    Update fitness scores for all individuals in the genetic table.
#
#     Parameters:
#         - gatable (np.ndarray): Genetic table containing gene values and fitness scores.
#         - profiles (np.ndarray): Generated profiles for all individuals.
#         - inputdata (np.ndarray): Input data for fitness calculation.
#         
#    Returns:
#        - np.ndarray: Updated genetic table with fitness scores.

def updateallfitnesses(gatable,profiles,inputdata):

    popsize=gatable.shape[0]
    indscore=gatable.shape[1]-1
    for n in range(popsize):
         gatable[n,indscore]=calculate_fitness(inputdata, profiles[:,n], gatable[n,3]*(f_max - f_min) + f_min, gatable[n,4]*(c_max - c_min) + c_min)
    return gatable


#    Generate children by crossover from parent individuals.
#
#    Parameters:
#        - parents (np.ndarray): Parent individuals.
#
#    Returns:
#        - np.ndarray or None: Children generated by crossover.

def generate_children(parents):
    
    size_parents = parents.shape
    numparents = size_parents[0]
    numchildren = popsize - numparents
    if numchildren % 2 !=0:
        print('numchildren must be even!')
        return None
    numpairs = int(numchildren/2)
    numcols = size_parents[1]
    #Using rank weighting for parent selection
    randnumbersparent = np.random.rand(numchildren)
    #each two consecutive rows mate
    parentindices = np.int64(np.floor((2*numparents+1-np.sqrt(4*numparents*(1+numparents)*(1-randnumbersparent)+1))/2))
    children = parents[parentindices,:]
    # perform crossover
    crossoverpoint = np.random.rand(numpairs)*3
    crossoverindex = np.int64(np.floor(crossoverpoint))
    crossovervalue = crossoverpoint - crossoverindex
    for n in range(numpairs):
        originalchild1 = children[2*n,:]
        originalchild2 = children[2*n+1,:]
        ind=crossoverindex[n]
        val=crossovervalue[n]
        newchild1 = np.hstack((originalchild1[0:ind],originalchild2[ind:]))
        newchild2 = np.hstack((originalchild2[0:ind],originalchild1[ind:]))
        newchild1[ind]= originalchild1[ind]*val+originalchild2[ind]*(1-val)
        newchild2[ind]= originalchild2[ind]*val+originalchild1[ind]*(1-val)
        newchild1[ind]=np.maximum(np.minimum(newchild1[ind],1),0)
        newchild2[ind]=np.maximum(np.minimum(newchild2[ind],1),0)
        #np.clip(newchild1[ind], 0, 1, out=newchild1[ind])
        #np.clip(newchild2[ind], 0, 1, out=newchild2[ind])
        children[2*n,:]=newchild1
        children[2*n+1,:]=newchild2
    return children


#    Apply mutations to the genetic table.
#
#    Parameters:
#        - gatable (np.ndarray): Genetic table containing gene values.
#        - numelites (int): Number of elite individuals.
#
#    Returns:
#        - np.ndarray: Genetic table with mutations applied.

def applymutations(gatable,numelites):
    
    shape_gatable = gatable.shape
    mutationhalfstepsize = 0.15
    mutationflag = np.less_equal(np.random.rand(shape_gatable[0],shape_gatable[1]),mutationrate)
    mutationvalues = np.random.uniform(-mutationhalfstepsize,mutationhalfstepsize,(shape_gatable[0],shape_gatable[1]))*mutationflag
    mutationvalues[0:numelites,:] = 0 #elite individuals are not mutated
    gatable = gatable + mutationvalues
    np.clip(gatable, 0, 1, out=gatable)    
    return gatable

# Read input experimental profile
q, input_data = np.loadtxt(sys.argv[1], unpack = 'True', usecols = (0, 1))
numq = len(q)

# Calculate weights of each q value for evaluation of fitness based on reference:
# Zijie Wu and Arthi Jayaraman, Macromolecules (2022), 55.24, 11076-11091
weights = np.zeros(numq)

for i in range(1, numq):
    weights[i] = np.log(q[i]/q[i-1])

weights[0] = weights[1]
weights = weights/np.sum(weights)

# Load XGBoost ML model that links structural features to computed 1D scattering profiles.
dirpath = './'
datapath = './'
outpath = './'
#dataset_file = dirpath + 'all_struc_features.txt'
model_file = dirpath + 'xgbmodel_all_trunc_spheres.json'
xgboost_model = xgb.Booster(model_file=model_file)

#Generate Initial Population
ipopsize=200
popsize=100
numgenes=5
gatable = np.random.rand(ipopsize,6)
currentprofiles = generateallprofiles(gatable)
gatable = updateallfitnesses(gatable, currentprofiles, input_data)
tableindices = gatable[:,numgenes].argsort() #sort by the descending fitness value
gatable = gatable[tableindices[0:popsize]]


# GA steps
numgens=200
numparents=30 # keep 30% of the population for mating
numelites=2
mutationrate=0.1

meanfitness = np.mean(gatable[:,5])
stddevfitness = np.std(gatable[:,5])
bestfitness = gatable[0,5]
worstfitness = gatable[-1,5]
diversitymetric = np.mean(np.sum((gatable-np.mean(gatable,axis=0))**2,axis=1))

#fitness scores initialization
fitness_scores = np.array([[0,meanfitness,stddevfitness,bestfitness,worstfitness]])

#Evolutionary process loop
struc_featurestable = genes_to_struc_features(gatable[:,0:5])
print('Generation: '+ str(0) +'. Best fitness: ' + str(bestfitness) + '. Average fitness: ' + str(meanfitness) + '.')

#Format for writing the output
fmt = "%.6f %.6f %.6f %.6f %.6f %.10f\n"
out = []

for currentgen in range(1,numgens+1):
    parents = gatable[0:numparents,:]
    children = generate_children(parents)
    gatable = np.vstack((parents,children))
    gatable = applymutations(gatable,numelites)
    currentprofiles = generateallprofiles(gatable)
    gatable = updateallfitnesses(gatable, currentprofiles, input_data)
    tableindices = gatable[:,numgenes].argsort() #sort by the descending fitness value
    gatable = gatable[tableindices]
    meanfitness = np.mean(gatable[:,5])
    stddevfitness = np.std(gatable[:,5])
    bestfitness = gatable[0,5]
    worstfitness = gatable[-1,5]
    diversitymetric = np.mean(np.sum((gatable-np.mean(gatable,axis=0))**2,axis=1))
    fitness_scores=np.append(fitness_scores,[[currentgen,meanfitness,stddevfitness,bestfitness,worstfitness]],axis=0)
    mutationrate = -np.log10(diversitymetric)*0.1
    struc_featurestable = genes_to_struc_features(gatable[:,0:5])

    if currentgen%10 == 0:
        print('Generation: '+ str(currentgen) +'. Best fitness: ' + str(bestfitness) + '. Average fitness: ' + str(meanfitness) + '.')


#   Printing all individuals once every 50 generations to a file.
#   Including the 3 structural features, the scale factor (f) and constant parameter (c) for every individual is also printed.
#   The last column provides the fitness of the individual.
    if currentgen%50 == 0:
        out = []
        for i in range(popsize):
            f_pop = gatable[i, 3]*(f_max - f_min) + f_min
            c_pop = gatable[i, 4]*(c_max - c_min) + c_min
            a = fmt % (struc_featurestable[i, 0], struc_featurestable[i, 1], struc_featurestable[i, 2], f_pop, c_pop, gatable[i, 5])
            out.append(a)

        open('All_Individuals_Gen_' + str(currentgen) + '.txt', 'w').writelines(out)

# Printing the range of variation for structural features as predicted by CREASE (Last generation)
# Here only the individuals that have a certain fitness (as defined by the user) are included in the range calculation

    if (currentgen==numgens):

        # If CREASE predicted MSE for best fit individual is higher than user defined MSE cut off then range of structural features cannot be output.
        if (mse_cut < gatable[0, 5]):
            print('Cannot find structural feature range for MSE cut off defined because best set of structural features has higher MSE than the cut off')

        else:
            for i in range(popsize):
                if (mse_cut < gatable[i, 5]):
                    cut_off_pop = int(i)
                    break

            # Printing the range of variation for all three structural features
            print('Minimum, Best Fit, and Maximum Values for mean volumetric radii in Angstroms:', np.min(struc_featurestable[:cut_off_pop, 0]), struc_featurestable[0, 0], np.max(struc_featurestable[:cut_off_pop, 0]))
            print('Minimum, Best Fit, and Maximum Values for std. volumetric radii in Angstroms:', np.min(np.multiply(struc_featurestable[:cut_off_pop, 1], struc_featurestable[:cut_off_pop, 0])), struc_featurestable[0, 1]*struc_featurestable[0, 0],  np.max(np.multiply(struc_featurestable[:cut_off_pop, 1], struc_featurestable[:cut_off_pop, 0])))
            print('Minimum, Best Fit, and Maximum Values for volume fraction:', np.min(struc_featurestable[:cut_off_pop, 2]), struc_featurestable[0, 2], np.max(struc_featurestable[:cut_off_pop, 2]))

