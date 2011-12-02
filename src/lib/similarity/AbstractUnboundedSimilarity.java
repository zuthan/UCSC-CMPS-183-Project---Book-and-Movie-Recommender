/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package lib.similarity;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.similarity.AbstractItemSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;
import org.apache.mahout.cf.taste.transforms.SimilarityTransform;

import com.google.common.base.Preconditions;
import scala.collection.mutable.ArrayBuffer;

/**
 * Abstract class to encapsulate Similarity functions that return an unbounded value (i.e not restricted to values in range [0,1])
 **/
abstract class AbstractUnboundedSimilarity extends AbstractItemSimilarity implements UserSimilarity {

  private PreferenceInferrer inferrer;
  private PreferenceTransform prefTransform;
  private SimilarityTransform similarityTransform;
  private int cachedNumItems;
  private int cachedNumUsers;
  private final RefreshHelper refreshHelper;

  /**
   * <p>
   * Creates an AbstractUnboundedSimilarity.
   * </p>
   */
  AbstractUnboundedSimilarity(final DataModel dataModel) throws TasteException {
    super(dataModel);

    this.cachedNumItems = dataModel.getNumItems();
    this.cachedNumUsers = dataModel.getNumUsers();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        cachedNumItems = dataModel.getNumItems();
        cachedNumUsers = dataModel.getNumUsers();
        return null;
      }
    });
  }

  final PreferenceInferrer getPreferenceInferrer() {
    return inferrer;
  }
  
  @Override
  public final void setPreferenceInferrer(PreferenceInferrer inferrer) {
    Preconditions.checkArgument(inferrer != null, "inferrer is null");
    refreshHelper.addDependency(inferrer);
    refreshHelper.removeDependency(this.inferrer);
    this.inferrer = inferrer;
  }
  
  public final PreferenceTransform getPrefTransform() {
    return prefTransform;
  }
  
  public final void setPrefTransform(PreferenceTransform prefTransform) {
    refreshHelper.addDependency(prefTransform);
    refreshHelper.removeDependency(this.prefTransform);
    this.prefTransform = prefTransform;
  }
  
  public final SimilarityTransform getSimilarityTransform() {
    return similarityTransform;
  }
  
  public final void setSimilarityTransform(SimilarityTransform similarityTransform) {
    refreshHelper.addDependency(similarityTransform);
    refreshHelper.removeDependency(this.similarityTransform);
    this.similarityTransform = similarityTransform;
  }

  abstract SimilarityAccumulator getSimilarityAccumulator();

  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    DataModel dataModel = getDataModel();
    PreferenceArray xPrefs = dataModel.getPreferencesFromUser(userID1);
    PreferenceArray yPrefs = dataModel.getPreferencesFromUser(userID2);
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();
    
    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }
    
    long xIndex = xPrefs.getItemID(0);
    long yIndex = yPrefs.getItemID(0);
    int xPrefIndex = 0;
    int yPrefIndex = 0;

    boolean hasInferrer = inferrer != null;
    boolean hasPrefTransform = prefTransform != null;

    SimilarityAccumulator accum = getSimilarityAccumulator();

    while (true) {
      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
      if (hasInferrer || compare == 0) {
        double x;
        double y;
        if (xIndex == yIndex) {
          // Both users expressed a preference for the item
          if (hasPrefTransform) {
            x = prefTransform.getTransformedValue(xPrefs.get(xPrefIndex));
            y = prefTransform.getTransformedValue(yPrefs.get(yPrefIndex));
          } else {
            x = xPrefs.getValue(xPrefIndex);
            y = yPrefs.getValue(yPrefIndex);
          }
        } else {
          // Only one user expressed a preference, but infer the other one's preference and tally
          // as if the other user expressed that preference
          if (compare < 0) {
            // X has a value; infer Y's
            x = hasPrefTransform
                ? prefTransform.getTransformedValue(xPrefs.get(xPrefIndex))
                : xPrefs.getValue(xPrefIndex);
            y = inferrer.inferPreference(userID2, xIndex);
          } else {
            // compare > 0
            // Y has a value; infer X's
            x = inferrer.inferPreference(userID1, yIndex);
            y = hasPrefTransform
                ? prefTransform.getTransformedValue(yPrefs.get(yPrefIndex))
                : yPrefs.getValue(yPrefIndex);
          }
        }

        accum.processPrefPair(x,y);

      }
      if (compare <= 0) {
        if (++xPrefIndex >= xLength) {
          if (hasInferrer) {
            // Must count other Ys; pretend next X is far away
            if (yIndex == Long.MAX_VALUE) {
              // ... but stop if both are done!
              break;
            }
            xIndex = Long.MAX_VALUE;
          } else {
            break;
          }
        } else {
          xIndex = xPrefs.getItemID(xPrefIndex);
        }
      }
      if (compare >= 0) {
        if (++yPrefIndex >= yLength) {
          if (hasInferrer) {
            // Must count other Xs; pretend next Y is far away            
            if (xIndex == Long.MAX_VALUE) {
              // ... but stop if both are done!
              break;
            }
            yIndex = Long.MAX_VALUE;
          } else {
            break;
          }
        } else {
          yIndex = yPrefs.getItemID(yPrefIndex);
        }
      }
    }
    
    double result;
    result = accum.computeResult();

    if (similarityTransform != null) {
      result = similarityTransform.transformSimilarity(userID1, userID2, result);
    }

    return result;
  }
  
  @Override
  public final double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    DataModel dataModel = getDataModel();
    PreferenceArray xPrefs = dataModel.getPreferencesForItem(itemID1);
    PreferenceArray yPrefs = dataModel.getPreferencesForItem(itemID2);
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();
    
    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }
    
    long xIndex = xPrefs.getUserID(0);
    long yIndex = yPrefs.getUserID(0);
    int xPrefIndex = 0;
    int yPrefIndex = 0;
    
    // No, pref inferrers and transforms don't appy here. I think.

      SimilarityAccumulator accum = getSimilarityAccumulator();

    while (true) {
      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
      if (compare == 0) {
        // Both users expressed a preference for the item
        double x = xPrefs.getValue(xPrefIndex);
        double y = yPrefs.getValue(yPrefIndex);

        accum.processPrefPair(x,y);
      }
      if (compare <= 0) {
        if (++xPrefIndex == xLength) {
          break;
        }
        xIndex = xPrefs.getUserID(xPrefIndex);
      }
      if (compare >= 0) {
        if (++yPrefIndex == yLength) {
          break;
        }
        yIndex = yPrefs.getUserID(yPrefIndex);
      }
    }

    double result;
    result = accum.computeResult();

    if (similarityTransform != null) {
      result = similarityTransform.transformSimilarity(itemID1, itemID2, result);
    }

    return result;
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = itemSimilarity(itemID1, itemID2s[i]);
    }
    return result;
  }

  @Override
  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    super.refresh(alreadyRefreshed);
    refreshHelper.refresh(alreadyRefreshed);
  }
  
  @Override
  public final String toString() {
    return this.getClass().getSimpleName() + "[dataModel:" + getDataModel() + ",inferrer:" + inferrer + ']';
  }
  
}
