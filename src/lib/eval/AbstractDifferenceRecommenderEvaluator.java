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

package lib.eval;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import scala.collection.mutable.ArrayBuffer;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Abstract superclass of a couple implementations, providing shared functionality.
 */
public abstract class AbstractDifferenceRecommenderEvaluator extends AbstractRecommenderEvaluator {

  protected abstract void processOneEstimate(float estimatedPreference, Preference realPref);

  protected void evaluateOneUser(Recommender recommender,
                                          long testUserID,
                                          PreferenceArray prefs,
                                          AtomicInteger noEstimateCounter) throws TasteException{
      for (Preference realPref : prefs) {
        float estimatedPreference = Float.NaN;
        try {
          estimatedPreference = recommender.estimatePreference(testUserID, realPref.getItemID());
        } catch (NoSuchUserException nsue) {
          // It's possible that an item exists in the test data but not training data in which case
          // NSEE will be thrown. Just ignore it and move on.
          log.info("User exists in test data but not training data: {}", testUserID);
        } catch (NoSuchItemException nsie) {
          log.info("Item exists in test data but not training data: {}", realPref.getItemID());
        }
        if (Float.isNaN(estimatedPreference)) {
          noEstimateCounter.incrementAndGet();
        } else {
          estimatedPreference = capEstimatedPreference(estimatedPreference);
          processOneEstimate(estimatedPreference, realPref);
        }
      }
  }

  
}
