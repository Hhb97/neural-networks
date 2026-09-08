package com.mv.demo;

/**
 * 模型评估器：计算多种回归评估指标。
 * <p>
 * 支持的指标：
 * <ul>
 *   <li>MSE (均方误差)</li>
 *   <li>MAE (平均绝对误差)</li>
 *   <li>R² (决定系数)</li>
 *   <li>CVRMSE (变异系数均方根误差)</li>
 *   <li>NMBE (归一化平均偏差误差)</li>
 * </ul>
 * </p>
 */
public class ModelEvaluator {

    /** 评估结果 */
    public static class Metrics {
        public final double mse;
        public final double mae;
        public final double r2;
        public final double cvrmse;
        public final double nmbe;

        public Metrics(double mse, double mae, double r2, double cvrmse, double nmbe) {
            this.mse = mse;
            this.mae = mae;
            this.r2 = r2;
            this.cvrmse = cvrmse;
            this.nmbe = nmbe;
        }

        @Override
        public String toString() {
            return String.format("MSE=%.4f, MAE=%.4f, R²=%.4f, CVRMSE=%.2f%%, NMBE=%.2f%%",
                    mse, mae, r2, cvrmse, nmbe);
        }
    }

    /**
     * 计算所有评估指标。
     *
     * @param actual    实际值数组
     * @param predicted 预测值数组
     * @return 评估指标对象
     */
    public static Metrics evaluate(double[] actual, double[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException(
                    "实际值与预测值长度不一致: " + actual.length + " vs " + predicted.length);
        }

        int n = actual.length;
        double sumSquaredError = 0.0;
        double sumAbsoluteError = 0.0;
        double sumBias = 0.0;
        double sumActual = 0.0;
        double sumActualSquared = 0.0;

        for (int i = 0; i < n; i++) {
            double error = actual[i] - predicted[i];
            sumSquaredError += error * error;
            sumAbsoluteError += Math.abs(error);
            sumBias += error;
            sumActual += actual[i];
            sumActualSquared += actual[i] * actual[i];
        }

        double mse = sumSquaredError / n;
        double mae = sumAbsoluteError / n;
        double meanActual = sumActual / n;

        // R² = 1 - SS_res / SS_tot
        double ssTot = 0.0;
        for (int i = 0; i < n; i++) {
            ssTot += (actual[i] - meanActual) * (actual[i] - meanActual);
        }
        double r2 = 1.0 - (sumSquaredError / ssTot);

        // CVRMSE = sqrt(MSE) / mean(actual) × 100
        double cvrmse = (Math.sqrt(mse) / Math.abs(meanActual)) * 100.0;

        // NMBE = sum(actual - predicted) / (n × mean(actual)) × 100
        double nmbe = (sumBias / (n * meanActual)) * 100.0;

        return new Metrics(mse, mae, r2, cvrmse, nmbe);
    }
}
