package lib;

import java.util.*;
/**
* Daniel Lemire
* A simple implementation of the weighted slope one
* algorithm in Java for item-based collaborative 
* filtering. 
* Assumes Java 1.5.
*
* See main function for example.
*
* June 1st 2006.
* Revised by Marco Ponzi on March 29th 2007
*/

public class SlopeOne {
  
  public static void main(String args[]){
    // this is my data base
    Map<UserId,Map<ItemId,Float>> data = new HashMap<UserId,Map<ItemId,Float>>();
    // items
    ItemId item1 = new ItemId("       candy");
    ItemId item2 = new ItemId("         dog");
    ItemId item3 = new ItemId("         cat");
    ItemId item4 = new ItemId("         war");
    ItemId item5 = new ItemId("strange food");
    
    mAllItems = new ItemId[]{item1, item2, item3, item4, item5};
    
    //I'm going to fill it in
    HashMap<ItemId,Float> user1 = new HashMap<ItemId,Float>();
    HashMap<ItemId,Float> user2 = new HashMap<ItemId,Float>();
    HashMap<ItemId,Float> user3 = new HashMap<ItemId,Float>();
    HashMap<ItemId,Float> user4 = new HashMap<ItemId,Float>();
    user1.put(item1,1.0f);
    user1.put(item2,0.5f);
    user1.put(item4,0.1f);
    data.put(new UserId("Bob"),user1);
    user2.put(item1,1.0f);
    user2.put(item3,0.5f);
    user2.put(item4,0.2f);
    data.put(new UserId("Jane"),user2);
    user3.put(item1,0.9f);
    user3.put(item2,0.4f);
    user3.put(item3,0.5f);
    user3.put(item4,0.1f);
    data.put(new UserId("Jo"),user3);    
    user4.put(item1,0.1f);
    //user4.put(item2,0.4f);
    //user4.put(item3,0.5f);
    user4.put(item4,1.0f);
    user4.put(item5,0.4f);
    data.put(new UserId("StrangeJo"),user4);    
    // next, I create my predictor engine
    SlopeOne so = new SlopeOne(data);
    System.out.println("Here's the data I have accumulated...");
    so.printData();
    // then, I'm going to test it out...
    HashMap<ItemId,Float> user = new HashMap<ItemId,Float>();
    System.out.println("Ok, now we predict...");
    user.put(item5,0.4f);
    System.out.println("Inputting...");
    SlopeOne.print(user);
    System.out.println("Getting...");
    SlopeOne.print(so.predict(user));
    //
    user.put(item4,0.2f);
    System.out.println("Inputting...");
    SlopeOne.print(user);
    System.out.println("Getting...");
    SlopeOne.print(so.predict(user));
  }
  
  Map<UserId,Map<ItemId,Float>> mData;
  Map<ItemId,Map<ItemId,Float>> mDiffMatrix;
  Map<ItemId,Map<ItemId,Integer>> mFreqMatrix;
  
  static ItemId[] mAllItems;
  
  public SlopeOne(Map<UserId,Map<ItemId,Float>> data) {
    mData = data;
    buildDiffMatrix();    
  }
  
  /**
  * Based on existing data, and using weights,
  * try to predict all missing ratings.
  * The trick to make this more scalable is to consider
  * only mDiffMatrix entries having a large  (>1) mFreqMatrix
  * entry.
  *
  * It will output the prediction 0 when no prediction is possible.
  */
  public Map<ItemId,Float> predict(Map<ItemId,Float> user) {
    HashMap<ItemId,Float> predictions = new HashMap<ItemId,Float>();
    HashMap<ItemId,Integer> frequencies = new HashMap<ItemId,Integer>();
    for (ItemId j : mDiffMatrix.keySet()) {
      frequencies.put(j,0);
      predictions.put(j,0.0f);
    }
    for (ItemId j : user.keySet()) {
      for (ItemId k : mDiffMatrix.keySet()) {
        try {
        float newval = ( mDiffMatrix.get(k).get(j).floatValue() + user.get(j).floatValue() ) * mFreqMatrix.get(k).get(j).intValue();
        predictions.put(k, predictions.get(k)+newval);
        frequencies.put(k, frequencies.get(k)+mFreqMatrix.get(k).get(j).intValue());
        } catch(NullPointerException e) {}
      }
    }
    HashMap<ItemId,Float> cleanpredictions = new HashMap<ItemId,Float>();
    for (ItemId j : predictions.keySet()) {
    	if (frequencies.get(j)>0) {
          cleanpredictions.put(j, predictions.get(j).floatValue()/frequencies.get(j).intValue());
    	}
    }
    for (ItemId j : user.keySet()) {
         cleanpredictions.put(j,user.get(j));
    }
    return cleanpredictions;
  }
  
  /**
  * Based on existing data, and not using weights,
  * try to predict all missing ratings.
  * The trick to make this more scalable is to consider
  * only mDiffMatrix entries having a large  (>1) mFreqMatrix
  * entry.
  */
  public Map<ItemId,Float> weightlesspredict(Map<ItemId,Float> user) {
    HashMap<ItemId,Float> predictions = new HashMap<ItemId,Float>();
    HashMap<ItemId,Integer> frequencies = new HashMap<ItemId,Integer>();
    for (ItemId j : mDiffMatrix.keySet()) {
      predictions.put(j,0.0f);
      frequencies.put(j,0);
    }
    for (ItemId j : user.keySet()) {
      for (ItemId k : mDiffMatrix.keySet()) {
        //System.out.println("Average diff between "+j+" and "+ k + " is "+mDiffMatrix.get(k).get(j).floatValue()+" with n = "+mFreqMatrix.get(k).get(j).floatValue());
        float newval = ( mDiffMatrix.get(k).get(j).floatValue() + user.get(j).floatValue() ) ;
        predictions.put(k, predictions.get(k)+newval);
      }
    }
    for (ItemId j : predictions.keySet()) {
        predictions.put(j, predictions.get(j).floatValue()/user.size());
    }
    for (ItemId j : user.keySet()) {
      predictions.put(j,user.get(j));
    }
    return predictions;
  }


  public void printData() {
	    for(UserId user : mData.keySet()) {
	      System.out.println(user);
	      print(mData.get(user));
	    }
	    for (int i=0; i<mAllItems.length; i++) {
	    	System.out.print("\n" + mAllItems[i] + ":");
	    	printMatrixes(mDiffMatrix.get(mAllItems[i]), mFreqMatrix.get(mAllItems[i]));
	    }
	  }

	private void printMatrixes(Map<ItemId,Float> ratings,
				Map<ItemId,Integer> frequencies) { 	
	    	for (int j=0; j<mAllItems.length; j++) {
	    		System.out.format("%10.3f", ratings.get(mAllItems[j]));
	    		System.out.print(" ");
	    		System.out.format("%10d", frequencies.get(mAllItems[j]));
	    	}
	    System.out.println();
	}
  
  public static void print(Map<ItemId,Float> user) {
    for (ItemId j : user.keySet()) {
      System.out.println(" "+ j+ " --> "+user.get(j).floatValue());
    }
  }
  
  public void buildDiffMatrix() {
    mDiffMatrix = new HashMap<ItemId,Map<ItemId,Float>>();
    mFreqMatrix = new HashMap<ItemId,Map<ItemId,Integer>>();
    // first iterate through users
    for(Map<ItemId,Float> user : mData.values()) {
      // then iterate through user data
      for(Map.Entry<ItemId,Float> entry: user.entrySet()) {
        if(!mDiffMatrix.containsKey(entry.getKey())) {
          mDiffMatrix.put(entry.getKey(), new HashMap<ItemId,Float>());
          mFreqMatrix.put(entry.getKey(), new HashMap<ItemId,Integer>());
        }
        for(Map.Entry<ItemId,Float> entry2: user.entrySet()) {
          int oldcount = 0;
          if(mFreqMatrix.get(entry.getKey()).containsKey(entry2.getKey()))
            oldcount = mFreqMatrix.get(entry.getKey()).get(entry2.getKey()).intValue();
          float olddiff = 0.0f;
          if(mDiffMatrix.get(entry.getKey()).containsKey(entry2.getKey()))
            olddiff = mDiffMatrix.get(entry.getKey()).get(entry2.getKey()).floatValue();
          float observeddiff = entry.getValue() - entry2.getValue();
          mFreqMatrix.get(entry.getKey()).put(entry2.getKey(),oldcount + 1);
          mDiffMatrix.get(entry.getKey()).put(entry2.getKey(),olddiff+observeddiff);          
        }
      }
    }
    for (ItemId j : mDiffMatrix.keySet()) {
      for (ItemId i : mDiffMatrix.get(j).keySet()) {
        float oldvalue = mDiffMatrix.get(j).get(i).floatValue();
        int count = mFreqMatrix.get(j).get(i).intValue();
        mDiffMatrix.get(j).put(i,oldvalue/count);
      }
    }
  }
}

class UserId  {
  String content;
  public UserId(String s) {
    content = s;
  }
  
  public int hashCode() { return content.hashCode();}
  public String toString() { return content; }
}
class ItemId  {
  String content;
  public ItemId(String s) {
    content = s;
  }
  public int hashCode() { return content.hashCode();}
  public String toString() { return content; }
}


