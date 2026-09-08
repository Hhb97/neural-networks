package com.mv.demo;

import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;

/**
 * OLS（普通最小二乘）回归模型，基于 Apache Commons Math 实现。
 * <p>
 * 使用 {@code OLSMultipleLinearRegression} 进行多元线性回归，
 * 对 M&V 能耗数据建立线性预测模型。
 * </p>
 */
public class OLSModel {

    private double[] regressionCoefficients;
    private OLSMultipleLinearRegression regression;

    /**
     * 训练 OLS 模型。
     *
     * @param x 训练特征矩阵 [n_samples × n_features]
     * @param y 目标值数组 [n_samples]
     */
    public void train(double[][] x, double[] y) {
        regression = new OLSMultipleLinearRegression();
        regression.newSampleData(y, x);
        regressionCoefficients = regression.estimateRegressionParameters();
    }

    /**
     * 使用训练好的模型进行预测。
     *
     * @param x 特征矩阵 [n_samples × n_features]
     * @return 预测值数组
     */
    public double[] predict(double[][] x) {
        double[] predictions = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            // y = intercept + β1*x1 + β2*x2 + β3*x3
            predictions[i] = regressionCoefficients[0]; // intercept
            for (int j = 0; j < x[i].length; j++) {
                predictions[i] += regressionCoefficients[j + 1] * x[i][j];
            }
        }
        return predictions;
    }

    /**
     * 获取回归系数。
     *
     * @return 系数数组，索引 0 为截距，其余为各特征系数
     */
    public double[] getCoefficients() {
        return regressionCoefficients;
    }

    /**
     * 获取 R² 值（仅训练集）。
     */
    public double getRSquare() {
        return regression.calculateRSquared();
    }

    /**
     * 生成回归公式的可读字符串。
     *
     * @return 公式描述
     */
    public String getFormula() {
        if (regressionCoefficients == null) {
            return "模型未训练";
        }
        String[] names = {"intercept", "temp", "humidity", "production"};
        StringBuilder sb = new StringBuilder();
        sb.append("y = ");
        for (int i = 0; i < regressionCoefficients.length; i++) {
            if (i > 0) {
                sb.append(" + ");
            }
            sb.append(String.format("%.4f", regressionCoefficients[i]));
            if (i > 0) {
                sb.append(" * ").append(names[i]);
            }
        }
        return sb.toString();
    }
}
