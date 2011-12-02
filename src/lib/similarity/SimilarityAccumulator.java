package lib.similarity;

public interface SimilarityAccumulator {
    void processPrefPair(Double u1Pref, Double u2Pref);

   /**
   * @return arbitrary similarity value, or {@link Double#NaN} if no similarity can be
   *         computed (e.g. when no items have been rated by both uesrs
   */
    Double computeResult();
}
