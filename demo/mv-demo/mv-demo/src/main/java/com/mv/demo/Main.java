package com.mv.demo;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

/**
 * M&V 能耗预测对比 —— 主入口。
 * <p>
 * 流程：数据生成 → 数据划分 → 模型训练 → 模型评估 → 控制台输出 → HTML 报告生成。
 * </p>
 */
public class Main {

    private static final int TRAIN_EPOCHS = 1000;
    private static final double LEARNING_RATE = 0.001;

    public static void main(String[] args) {
        System.out.println("========== M&V 能耗预测对比 ==========\n");

        // ===== 1. 生成数据 =====
        System.out.println("[1/6] 生成模拟数据...");
        DataGenerator generator = new DataGenerator();
        List<DataGenerator.DataPoint> allData = generator.generate();
        int totalSize = allData.size();
        int trainSize = totalSize / 2;
        int testSize = totalSize - trainSize;

        System.out.printf(Locale.US, "数据量: %d 条 (训练: %d, 测试: %d)%n", totalSize, trainSize, testSize);
        System.out.println("特征变量: temperature, humidity, production\n");

        // ===== 2. 提取特征和标签 =====
        double[][] xAll = new double[totalSize][3];
        double[] yAll = new double[totalSize];
        for (int i = 0; i < totalSize; i++) {
            DataGenerator.DataPoint dp = allData.get(i);
            xAll[i][0] = dp.temperature;
            xAll[i][1] = dp.humidity;
            xAll[i][2] = dp.production;
            yAll[i] = dp.energyConsumption;
        }

        // 划分训练集和测试集
        double[][] xTrain = new double[trainSize][3];
        double[] yTrain = new double[trainSize];
        double[][] xTest = new double[testSize][3];
        double[] yTest = new double[testSize];

        System.arraycopy(xAll, 0, xTrain, 0, trainSize);
        System.arraycopy(yAll, 0, yTrain, 0, trainSize);
        System.arraycopy(xAll, trainSize, xTest, 0, testSize);
        System.arraycopy(yAll, trainSize, yTest, 0, testSize);

        // ===== 3. 训练 OLS 模型 =====
        System.out.println("[2/6] 训练 OLS 模型...");
        OLSModel olsModel = new OLSModel();
        olsModel.train(xTrain, yTrain);
        double[] olsTrainPred = olsModel.predict(xTrain);
        double[] olsTestPred = olsModel.predict(xTest);

        // ===== 4. 训练神经网络模型 =====
        System.out.println("\n[3/6] 训练神经网络模型...");
        NNModel nnModel = new NNModel();
        nnModel.train(xTrain, yTrain, TRAIN_EPOCHS, LEARNING_RATE);
        double[] nnTrainPred = nnModel.predict(xTrain);
        double[] nnTestPred = nnModel.predict(xTest);

        // ===== 5. 评估 =====
        System.out.println("\n[4/6] 计算评估指标...");
        ModelEvaluator.Metrics olsTrainMetrics = ModelEvaluator.evaluate(yTrain, olsTrainPred);
        ModelEvaluator.Metrics olsTestMetrics = ModelEvaluator.evaluate(yTest, olsTestPred);
        ModelEvaluator.Metrics nnTrainMetrics = ModelEvaluator.evaluate(yTrain, nnTrainPred);
        ModelEvaluator.Metrics nnTestMetrics = ModelEvaluator.evaluate(yTest, nnTestPred);

        // ===== 6. 控制台输出 =====
        System.out.println("\n[5/6] 输出对比结果...\n");

        System.out.println("【OLS 模型】");
        System.out.println("  公式: " + olsModel.getFormula());
        System.out.printf(Locale.US, "  训练集 R²: %.4f%n", olsTrainMetrics.r2);
        System.out.printf(Locale.US, "  测试集 R²: %.4f%n", olsTestMetrics.r2);
        System.out.printf(Locale.US, "  测试集 CVRMSE: %.2f%%%n", olsTestMetrics.cvrmse);
        System.out.printf(Locale.US, "  测试集 NMBE: %.2f%%%n", olsTestMetrics.nmbe);

        System.out.println("\n【神经网络模型】");
        System.out.println("  结构: 3 → 64(ReLU) → 32(ReLU) → 1");
        System.out.printf(Locale.US, "  参数数量: %,d 个%n", nnModel.getParamCount());
        System.out.printf(Locale.US, "  训练 Epochs: %d%n", TRAIN_EPOCHS);
        System.out.printf(Locale.US, "  训练集 R²: %.4f%n", nnTrainMetrics.r2);
        System.out.printf(Locale.US, "  测试集 R²: %.4f%n", nnTestMetrics.r2);
        System.out.printf(Locale.US, "  测试集 CVRMSE: %.2f%%%n", nnTestMetrics.cvrmse);
        System.out.printf(Locale.US, "  测试集 NMBE: %.2f%%%n", nnTestMetrics.nmbe);

        double r2Improve = ((nnTestMetrics.r2 - olsTestMetrics.r2) / olsTestMetrics.r2) * 100.0;
        double cvrmseReduce = ((olsTestMetrics.cvrmse - nnTestMetrics.cvrmse) / olsTestMetrics.cvrmse) * 100.0;

        System.out.println("\n【对比总结】");
        System.out.printf(Locale.US, "  精度提升: R² 提升 %.1f%%%n", r2Improve);
        System.out.printf(Locale.US, "  误差降低: CVRMSE 降低 %.1f%%%n", cvrmseReduce);
        String conclusion = nnTestMetrics.r2 > olsTestMetrics.r2
                ? "神经网络效果显著优于 OLS"
                : "OLS 效果优于或接近神经网络";
        System.out.println("  结论: " + conclusion);

        // ===== 7. 生成 HTML 报告 =====
        System.out.println("\n[6/6] 生成 HTML 报告...");
        try {
            HtmlReporter reporter = new HtmlReporter();
            String outputPath = "mv_comparison_report.html";
            reporter.generate(outputPath,
                    yTest, olsTestPred, nnTestPred,
                    olsTestMetrics, nnTestMetrics,
                    olsModel.getFormula(), nnModel.getParamCount(),
                    trainSize, testSize);
            System.out.println("📄 HTML 报告已生成: ./" + outputPath);
        } catch (IOException e) {
            System.err.println("❌ HTML 报告生成失败: " + e.getMessage());
        }

        // 清理资源
        nnModel.close();

        System.out.println("\n✅ 对比完成！");
    }
}
