package com.mv.demo;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslatorContext;

/**
 * 基于 DJL (Deep Java Library) 的多层感知器 (MLP) 神经网络模型。
 * <p>
 * 网络结构：3 → 64(ReLU) → 32(ReLU) → 1
 * 用于 M&V 能耗预测中的非线性回归任务。
 * </p>
 */
public class NNModel {

    private Model model;
    private NDManager manager;
    private double[] featureMin;
    private double[] featureMax;
    private double labelMin;
    private double labelMax;
    private int totalParams;

    /**
     * 训练神经网络模型。
     *
     * @param x       训练特征矩阵 [n_samples × 3]
     * @param y       目标值数组 [n_samples]
     * @param epochs  训练轮数
     * @param lr      学习率
     */
    public void train(double[][] x, double[] y, int epochs, double lr) {
        int nSamples = x.length;
        int nFeatures = x[0].length;

        // 计算归一化参数
        computeNormalizationParams(x, y);

        // 归一化数据
        double[][] xNorm = normalizeFeatures(x);
        double[] yNorm = normalizeLabels(y);

        // 创建 NDManager
        manager = NDManager.newBaseManager();

        // 创建 NDArray
        NDArray xArray = manager.create(flattenArray(xNorm), new Shape(nSamples, nFeatures));
        NDArray yArray = manager.create(yNorm, new Shape(nSamples, 1));

        // 构建 MLP 网络: 3 → 64(ReLU) → 32(ReLU) → 1
        SequentialBlock mlp = new SequentialBlock();
        mlp.add(Linear.builder().setUnits(64).build());
        mlp.add(Activation.reluBlock());
        mlp.add(Linear.builder().setUnits(32).build());
        mlp.add(Activation.reluBlock());
        mlp.add(Linear.builder().setUnits(1).build());

        // 计算参数数量
        totalParams = (3 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1);

        // 创建模型
        try {
            model = Model.newInstance("mv-mlp");
            model.setBlock(mlp);

            // 配置训练器
            Loss loss = Loss.l2Loss();
            Optimizer optimizer = Optimizer.adam()
                    .optLearningRateTracker(Tracker.fixed((float) lr))
                    .optBeta1(0.9f)
                    .optBeta2(0.999f)
                    .optEpsilon(1e-8f)
                    .build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(optimizer)
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, nFeatures));

                // 训练循环
                for (int epoch = 1; epoch <= epochs; epoch++) {
                    try (ai.djl.training.GradientCollector gc = trainer.newGradientCollector()) {
                        NDArray pred = trainer.forward(new NDList(xArray)).singletonOrThrow();
                        NDArray lossValue = loss.evaluate(new NDList(yArray), new NDList(pred));
                        gc.backward(lossValue);

                        if (epoch % 100 == 0 || epoch == 1) {
                            float lv = lossValue.toType(ai.djl.ndarray.types.DataType.FLOAT32, false).getFloat();
                            System.out.printf("    Epoch %4d/%d, Loss: %.6f%n", epoch, epochs, lv);
                        }
                    }
                    trainer.step();
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("神经网络训练失败: " + e.getMessage(), e);
        }
    }

    /**
     * 使用训练好的模型进行预测。
     *
     * @param x 特征矩阵 [n_samples × 3]
     * @return 预测值数组（原始尺度）
     */
    public double[] predict(double[][] x) {
        if (model == null || manager == null) {
            throw new IllegalStateException("模型未训练，请先调用 train() 方法");
        }

        int nSamples = x.length;
        int nFeatures = x[0].length;

        double[][] xNorm = normalizeFeatures(x);
        NDArray xArray = manager.create(flattenArray(xNorm), new Shape(nSamples, nFeatures));

        try (Trainer trainer = model.newTrainer(
                new DefaultTrainingConfig(Loss.l2Loss()))) {
            trainer.initialize(new Shape(1, nFeatures));
            NDArray pred = trainer.forward(new NDList(xArray)).singletonOrThrow();

            float[] rawPred = pred.toType(ai.djl.ndarray.types.DataType.FLOAT32, false).toFloatArray();
            double[] predictions = new double[nSamples];
            for (int i = 0; i < nSamples; i++) {
                predictions[i] = denormalizeLabel(rawPred[i]);
            }
            return predictions;
        } catch (Exception e) {
            throw new RuntimeException("神经网络预测失败: " + e.getMessage(), e);
        }
    }

    /**
     * 获取模型参数数量。
     */
    public int getParamCount() {
        return totalParams;
    }

    /**
     * 关闭资源。
     */
    public void close() {
        if (manager != null) {
            manager.close();
        }
        if (model != null) {
            model.close();
        }
    }

    // ==================== 归一化辅助方法 ====================

    private void computeNormalizationParams(double[][] x, double[] y) {
        int nFeatures = x[0].length;
        featureMin = new double[nFeatures];
        featureMax = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++) {
            featureMin[j] = Double.MAX_VALUE;
            featureMax[j] = -Double.MAX_VALUE;
            for (double[] row : x) {
                featureMin[j] = Math.min(featureMin[j], row[j]);
                featureMax[j] = Math.max(featureMax[j], row[j]);
            }
        }

        labelMin = Double.MAX_VALUE;
        labelMax = -Double.MAX_VALUE;
        for (double v : y) {
            labelMin = Math.min(labelMin, v);
            labelMax = Math.max(labelMax, v);
        }
    }

    private double[][] normalizeFeatures(double[][] x) {
        double[][] norm = new double[x.length][x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                double range = featureMax[j] - featureMin[j];
                norm[i][j] = (range == 0) ? 0.0 : (x[i][j] - featureMin[j]) / range;
            }
        }
        return norm;
    }

    private double[] normalizeLabels(double[] y) {
        double[] norm = new double[y.length];
        double range = labelMax - labelMin;
        for (int i = 0; i < y.length; i++) {
            norm[i] = (range == 0) ? 0.0 : (y[i] - labelMin) / range;
        }
        return norm;
    }

    private double denormalizeLabel(double normVal) {
        return normVal * (labelMax - labelMin) + labelMin;
    }

    private float[] flattenArray(double[][] arr) {
        float[] flat = new float[arr.length * arr[0].length];
        int idx = 0;
        for (double[] row : arr) {
            for (double v : row) {
                flat[idx++] = (float) v;
            }
        }
        return flat;
    }
}
