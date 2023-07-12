import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
def best_arm_finder(rewards,counts,trial,n_arms=12,c=1):
    """This function find best arm for UCB
    Args:
      rewards: list of rewards, where the reward for each arm is the average reward after trial-1
      counts: List of Number of time a arm is played after trial-1
      trial: current trial
      n_arms: Number of arms in UCB
      c:UCB constant
    Returns:
      a Best arm after trial round
    """
    best_arm = -1
    best_ucb = -float('inf')
    for arm in range(n_arms):
        ucb_val = rewards[arm] / counts[arm] + c*np.sqrt(np.log(trial+1)/counts[arm])
        if ucb_val > best_ucb:
          best_arm = arm
          best_ucb = ucb_val
    return best_arm

def reward_cost_yPred_generator(sample,thresh,arm,lembda,o,is_best_arm):
    """This Function find reward,cost,prediction for a sample

    Args:
        sample (array): list of confidence for each exit
        thresh (float): Threshhold 
        arm (integer): optimal exit at particular trial
        lembda (float): battery deplication rate
        o (float): offloading cost
        is_best_arm: if best arm then True otherwise False

    Returns:
        reward: reward of particular trial
        cost: cost of particular trial
        pred:prediction of the trial 
        co: sample is offloaded or not   
    """
    cost=None
    pred=None
    co=False
    if arm<11:
        if max(sample[arm])>=thresh:
            reward=max(sample[arm])-lembda*((arm+1))
            cost=lembda*((arm+1))
            pred=np.argmax(sample[arm])
        else:
            reward=max(sample[11])-o-lembda*((arm+1))
            cost=o+lembda*((arm+1))
            pred=np.argmax(sample[11])
            co=True
    else:
        reward=max(sample[arm])-lembda*((arm+1))
        cost=lembda*((arm+1))
        pred=np.argmax(sample[11])
    return reward,cost,pred,co

def accuracy_generate(y,y_pred):
    """This function generate accuracy

    Args:
        y (array): target
        y_pred (_type_): predicted array

    Returns:
      accuracy of dataset
    """
    return np.sum(np.array(y_pred)==y)/len(y)
def UCB(df,threshhold,lembda,o,n_arms=12,n_epoch=5,c=1):
    """
    This function implements the Upper Confidence Bound (UCB) algorithm.

    Args:
        df: dataframe
        threshhold: threshhold value
        lembda :    battery deplication rate
        O:          Ofloading cost
        n_arms:     The number of arms.
        n_epoch:    The number of epoch.
        c:          costant of ucb

    Returns:
        optimum layer of Exit
        average accuracy
        average number of sample offloaded
        average cost after n_epoch
        """
    rewards_list_final=[]
    accuracy_list=[]
    cost_list=[]
    sample_offloaded=[]
    for _ in range(n_epoch):
        data=df.sample(frac=1).to_numpy()
        x=data[:,:-1]
        y=data[:,-1]
        #intialise required list
        rewards_list=[]
        count_list=[0 for i in range(n_arms)]
        prediction=[]
        cost=0
        offload=0
        #initialise reward
        for i in range(n_arms):
            reward,__,__,__=reward_cost_yPred_generator(x[i],threshhold,i,lembda,o,is_best_arm=False)
            rewards_list.append(reward)
            count_list[i]+=1
        for trial in range(df.shape[0]):
            best_arm=best_arm_finder(rewards_list,count_list,trial,n_arms=12,c=1)
            reward,cost_,y_pred,co=reward_cost_yPred_generator(x[trial],threshhold,best_arm,lembda,o,is_best_arm=True)
            rewards_list[best_arm]+=reward
            count_list[best_arm]+=1
            cost+=cost_
            prediction.append(y_pred)
            offload+=co
        rewards_list_final.append(rewards_list)
        cost_list.append(cost)
        accuracy_list.append(accuracy_generate(y,np.array(prediction)))
        sample_offloaded.append(offload)
    return (np.argmax(np.mean(rewards_list_final,axis=0)),np.mean(accuracy_list),np.mean(sample_offloaded),np.mean(cost_list))



df_results_dict=pd.read_pickle('/home/divya/updated_code/Dataset/confidence_label_dict.pkl')
dataset_name=['imdb','scitail','yelp','qqp','snli']
o_value = [0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
threshhold = {'imdb':0.7,'scitail':0.85,'yelp':0.7,'qqp':0.75,'snli':0.75}

lembda=1/10
result_dict={}
for dataset in dataset_name:
    df=df_results_dict[dataset]
    result={}
    co=0
    for offload_cost in tqdm(o_value):
        result[offload_cost]=tuple(UCB(df,threshhold[dataset],lembda,offload_cost,n_arms=12,n_epoch=5,c=1))
    result_dict[dataset]=result

with open("/home/divya/updated_code/Dataset/ucb_result_without_using_side_information.pkl",'wb') as file:
    pickle.dump(result_dict,file)


#best_arm,accuracy,offloaded,cost=UCB(df_results_dict["df_"+sys.argv[1]],threshhold[sys.argv[1]],lembda,float(sys.argv[2]),n_arms=12,n_epoch=5,c=1)
#print("optimum arm =",best_arm,"\n average accuracy =",accuracy,"\n average number of sample offloaded =",np.floor(offloaded),"\n average cost=",cost,'\n threshhold for the '+sys.argv[1]+"=",threshhold[sys.argv[1]])