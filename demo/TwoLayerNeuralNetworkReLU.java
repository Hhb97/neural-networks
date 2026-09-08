package com.xxl.job.admin;

import java.util.Arrays;
import java.util.Random;

/**
 * 两层神经网络 + ReLU 激活函数
 * 
 * 结构：
 *   输入层 (n个特征) 
 *     → 隐藏层: z1 = X × W1 + b1,  a1 = ReLU(z1)
 *     → 输出层: z2 = a1 × W2 + b2,  ŷ = z2 (回归任务，无激活)
 * 
 * 等价于：ŷ = ReLU(X × W1 + b1) × W2 + b2
 */
public class TwoLayerNeuralNetworkReLU {

    private int inputSize;      // 输入特征数
    private int hiddenSize;     // 隐藏层神经元数
    private int outputSize;     // 输出维度（回归任务固定为1）
    
    private double[][] W1;      // 输入→隐藏层权重 (inputSize × hiddenSize)
    private double[] b1;        // 隐藏层偏置 (hiddenSize)
    private double[][] W2;      // 隐藏层→输出层权重 (hiddenSize × outputSize)
    private double[] b2;        // 输出层偏置 (outputSize)
    
    private double learningRate;
    private int epochs;
    private double lambda;  // L2 正则化系数
    
    // 归一化参数
    private double[] featureMean;
    private double[] featureStd;
    private double targetMean;
    private double targetStd;
    
    private static final Random RANDOM = new Random(42);
    
    // ==================== 构造函数 ====================
    public TwoLayerNeuralNetworkReLU(int inputSize, int hiddenSize, double learningRate, int epochs) {
        this(inputSize, hiddenSize, learningRate, epochs, 0.001);
    }
    
    public TwoLayerNeuralNetworkReLU(int inputSize, int hiddenSize, double learningRate, int epochs, double lambda) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = 1;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.lambda = lambda;
        initializeWeights();
    }
    
    // ==================== 权重初始化 (He初始化) ====================
    private void initializeWeights() {
        // W1: inputSize × hiddenSize
        W1 = new double[inputSize][hiddenSize];
        double std1 = Math.sqrt(2.0 / inputSize);
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W1[i][j] = RANDOM.nextGaussian() * std1;
            }
        }
        b1 = new double[hiddenSize];
        
        // W2: hiddenSize × outputSize
        W2 = new double[hiddenSize][outputSize];
        double std2 = Math.sqrt(2.0 / hiddenSize);
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                W2[i][j] = RANDOM.nextGaussian() * std2;
            }
        }
        b2 = new double[outputSize];
    }
    
    // ==================== ReLU 激活函数 ====================
    private double relu(double x) {
        return Math.max(0, x);
    }
    
    // ReLU 的导数（用于反向传播）
    private double reluDerivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }
    
    // 对向量应用 ReLU
    private double[] reluVector(double[] z) {
        double[] a = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            a[i] = relu(z[i]);
        }
        return a;
    }
    
    // ==================== 前向传播 ====================
    /**
     * 前向传播（返回所有中间结果，用于反向传播）
     */
    private ForwardResult forward(double[] x) {
        // 1. 隐藏层：z1 = x × W1 + b1
        double[] z1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) {
                sum += x[i] * W1[i][j];
            }
            z1[j] = sum + b1[j];
        }
        
        // 2. 隐藏层激活：a1 = ReLU(z1)
        double[] a1 = reluVector(z1);
        
        // 3. 输出层：z2 = a1 × W2 + b2
        double z2 = 0;
        for (int j = 0; j < hiddenSize; j++) {
            z2 += a1[j] * W2[j][0];
        }
        z2 += b2[0];
        
        // 4. 输出：ŷ = z2 (回归任务无激活)
        double y_pred = z2;
        
        return new ForwardResult(z1, a1, z2, y_pred);
    }
    
    /**
     * 前向传播结果封装
     */
    private static class ForwardResult {
        double[] z1;      // 隐藏层线性输出
        double[] a1;      // 隐藏层激活输出 (ReLU后)
        double z2;        // 输出层线性输出
        double y_pred;    // 最终预测值
        
        ForwardResult(double[] z1, double[] a1, double z2, double y_pred) {
            this.z1 = z1;
            this.a1 = a1;
            this.z2 = z2;
            this.y_pred = y_pred;
        }
    }
    
    // ==================== 反向传播 ====================
    /**
     * 反向传播（链式法则），更新权重
     */
    private void backward(double[] x, double y_true, ForwardResult fr) {
        int m = 1; // 单样本训练，批量训练需累加
        
        // ====== 输出层梯度 ======
        // dL/dŷ = ŷ - y   (MSE损失: L = 0.5*(ŷ-y)²)
        double dL_dy_pred = fr.y_pred - y_true;
        
        // dL/dz2 = dL/dŷ * dŷ/dz2 = dL/dŷ (因为 ŷ = z2)
        double dL_dz2 = dL_dy_pred;
        
        // dL/dW2 = a1^T × dL/dz2
        double[] dL_dW2 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            dL_dW2[j] = fr.a1[j] * dL_dz2;
        }
        
        // dL/db2 = dL/dz2
        double dL_db2 = dL_dz2;
        
        // ====== 传回隐藏层 ======
        // dL/da1 = dL/dz2 × W2^T
        double[] dL_da1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            dL_da1[j] = dL_dz2 * W2[j][0];
        }
        
        // dL/dz1 = dL/da1 ⊙ ReLU'(z1)  (逐元素乘)
        double[] dL_dz1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            dL_dz1[j] = dL_da1[j] * reluDerivative(fr.z1[j]);
        }
        
        // dL/dW1 = x^T × dL/dz1
        double[][] dL_dW1 = new double[inputSize][hiddenSize];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                dL_dW1[i][j] = x[i] * dL_dz1[j];
            }
        }
        
        // dL/db1 = dL/dz1
        double[] dL_db1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            dL_db1[j] = dL_dz1[j];
        }
        
        // ====== 梯度下降更新（含L2正则化）======
        // W_new = W_old - lr × (gradient + lambda × W_old)
        // 更新 W2, b2
        for (int j = 0; j < hiddenSize; j++) {
            W2[j][0] -= learningRate * (dL_dW2[j] + lambda * W2[j][0]);
        }
        b2[0] -= learningRate * dL_db2;
        
        // 更新 W1, b1
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W1[i][j] -= learningRate * (dL_dW1[i][j] + lambda * W1[i][j]);
            }
        }
        for (int j = 0; j < hiddenSize; j++) {
            b1[j] -= learningRate * dL_db1[j];
        }
    }
    
    // ==================== 训练 ====================
    /**
     * 训练模型
     * @param X 训练数据 (m × inputSize)
     * @param y 训练标签 (m)
     */
    public void train(double[][] X, double[] y) {
        int m = X.length;
        
        // 1. 数据标准化
        normalizeData(X, y);
        double[][] XNorm = normalizeX(X);
        double[] yNorm = normalizeY(y);
        
        System.out.println("========== 开始训练 ==========");
        System.out.println("样本数: " + m + ", 特征数: " + inputSize + ", 隐藏层神经元: " + hiddenSize);
        System.out.println("学习率: " + learningRate + ", 迭代次数: " + epochs + ", L2正则化: " + lambda);
        System.out.println();
        
        // 2. 训练循环（含早停机制）
        double bestLoss = Double.MAX_VALUE;
        int patience = 500;     // 容忍多少轮不改善
        int wait = 0;
        double[][] bestW1 = null, bestW2 = null;
        double[] bestB1 = null, bestB2 = null;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            
            // 遍历每个样本（随机梯度下降）
            for (int i = 0; i < m; i++) {
                // 前向传播
                ForwardResult fr = forward(XNorm[i]);
                
                // 计算损失 (MSE)
                double loss = 0.5 * Math.pow(fr.y_pred - yNorm[i], 2);
                totalLoss += loss;
                
                // 反向传播 + 梯度下降
                backward(XNorm[i], yNorm[i], fr);
            }
            
            double avgLoss = totalLoss / m;
            
            // 早停：记录最佳权重
            if (avgLoss < bestLoss) {
                bestLoss = avgLoss;
                wait = 0;
                bestW1 = copyArray(W1);
                bestW2 = copyArray(W2);
                bestB1 = b1.clone();
                bestB2 = b2.clone();
            } else {
                wait++;
            }
            
            // 每 100 轮打印一次损失
            if (epoch % 100 == 0 || epoch == epochs - 1) {
                System.out.printf("Epoch %d, Loss: %.6f%s\n", epoch, avgLoss, 
                    wait > 0 ? " (patience: " + wait + "/" + patience + ")" : " ★ best");
            }
            
            // 早停触发
            if (wait >= patience) {
                System.out.printf("早停触发! 在 Epoch %d 恢复最佳权重 (Loss: %.6f)\n", epoch, bestLoss);
                W1 = bestW1; W2 = bestW2; b1 = bestB1; b2 = bestB2;
                break;
            }
        }
        
        // 如果没有触发早停，也恢复最佳权重
        if (bestW1 != null && wait > 0) {
            W1 = bestW1; W2 = bestW2; b1 = bestB1; b2 = bestB2;
        }
        
        System.out.println("\n========== 训练完成 ==========");
        printWeightsAndFormula();
    }
    
    private double[][] copyArray(double[][] arr) {
        double[][] copy = new double[arr.length][];
        for (int i = 0; i < arr.length; i++) {
            copy[i] = arr[i].clone();
        }
        return copy;
    }
    
    // ==================== 预测 ====================
    /**
     * 预测（输入原始数据，输出原始尺度的预测值）
     */
    public double[] predict(double[][] X) {
        int m = X.length;
        double[][] XNorm = normalizeX(X);
        double[] predictions = new double[m];
        
        for (int i = 0; i < m; i++) {
            ForwardResult fr = forward(XNorm[i]);
            // 反标准化
            predictions[i] = fr.y_pred * targetStd + targetMean;
        }
        return predictions;
    }
    
    /**
     * 单样本预测
     */
    public double predict(double[] x) {
        double[] xNorm = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            xNorm[i] = (x[i] - featureMean[i]) / featureStd[i];
        }
        ForwardResult fr = forward(xNorm);
        return fr.y_pred * targetStd + targetMean;
    }
    
    // ==================== 数据标准化 ====================
    private void normalizeData(double[][] X, double[] y) {
        int m = X.length;
        int n = X[0].length;
        
        featureMean = new double[n];
        featureStd = new double[n];
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) sum += X[i][j];
            featureMean[j] = sum / m;
            
            double sumSq = 0;
            for (int i = 0; i < m; i++) sumSq += Math.pow(X[i][j] - featureMean[j], 2);
            featureStd[j] = Math.sqrt(sumSq / m);
            if (featureStd[j] == 0) featureStd[j] = 1;
        }
        
        double sumY = 0;
        for (double v : y) sumY += v;
        targetMean = sumY / m;
        
        double sumYSq = 0;
        for (double v : y) sumYSq += Math.pow(v - targetMean, 2);
        targetStd = Math.sqrt(sumYSq / m);
        if (targetStd == 0) targetStd = 1;
    }
    
    private double[][] normalizeX(double[][] X) {
        int m = X.length;
        int n = X[0].length;
        double[][] XNorm = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                XNorm[i][j] = (X[i][j] - featureMean[j]) / featureStd[j];
            }
        }
        return XNorm;
    }
    
    private double[] normalizeY(double[] y) {
        double[] yNorm = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            yNorm[i] = (y[i] - targetMean) / targetStd;
        }
        return yNorm;
    }
    
    // ==================== 输出权重和公式 ====================
    /**
     * 打印权重矩阵和最终公式
     */
    public void printWeightsAndFormula() {
        System.out.println("\n========== 权重矩阵 ==========");
        
        System.out.println("\n【W1】输入→隐藏层 (" + inputSize + " × " + hiddenSize + "):");
        for (int i = 0; i < inputSize; i++) {
            System.out.print("  W1[" + i + "] = [");
            for (int j = 0; j < hiddenSize; j++) {
                System.out.printf("%8.4f", W1[i][j]);
                if (j < hiddenSize - 1) System.out.print(", ");
            }
            System.out.println(" ]");
        }
        
        System.out.println("\n【b1】隐藏层偏置 (" + hiddenSize + "):");
        System.out.print("  b1 = [");
        for (int j = 0; j < hiddenSize; j++) {
            System.out.printf("%8.4f", b1[j]);
            if (j < hiddenSize - 1) System.out.print(", ");
        }
        System.out.println(" ]");
        
        System.out.println("\n【W2】隐藏层→输出 (" + hiddenSize + " × 1):");
        System.out.print("  W2 = [");
        for (int j = 0; j < hiddenSize; j++) {
            System.out.printf("%8.4f", W2[j][0]);
            if (j < hiddenSize - 1) System.out.print(", ");
        }
        System.out.println(" ]^T");
        
        System.out.println("\n【b2】输出层偏置:");
        System.out.printf("  b2 = %.4f\n", b2[0]);
        
        // 打印最终公式
        System.out.println("\n========== 最终公式 ==========");
        System.out.println("ŷ = ReLU(X × W1 + b1) × W2 + b2");
        System.out.println("\n展开形式:");
        
        // 隐藏层每个神经元的公式
        for (int j = 0; j < hiddenSize; j++) {
            System.out.printf("  h%d = ReLU(", j + 1);
            for (int i = 0; i < inputSize; i++) {
                System.out.printf("%.4f·x%d", W1[i][j], i + 1);
                if (i < inputSize - 1) System.out.print(" + ");
            }
            System.out.printf(" + %.4f)\n", b1[j]);
        }
        
        // 输出层公式
        System.out.print("  ŷ = ");
        for (int j = 0; j < hiddenSize; j++) {
            System.out.printf("%.4f·h%d", W2[j][0], j + 1);
            if (j < hiddenSize - 1) System.out.print(" + ");
        }
        System.out.printf(" + %.4f\n", b2[0]);
        
        // ====== 等效线性公式（类似最小二乘法 y = k1*x1 + k2*x2 + ... + b）======
        // 通过正向传播计算每个隐藏神经元在 x=0 处的激活状态，
        // 估算在均值附近的等效线性系数
        System.out.println("\n========== 等效线性公式 (近似) ==========");
        System.out.println("注意: 由于 ReLU 非线性，以下为在样本均值附近的近似线性表达");
        
        // 计算每个隐藏神经元在均值输入下的状态（激活 or 未激活）
        double[] xMean = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            xMean[i] = featureMean[i];
        }
        // 标准化后的均值输入（全0）
        double[] xMeanNorm = new double[inputSize];
        double[] z1Mean = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) {
                sum += xMeanNorm[i] * W1[i][j];
            }
            z1Mean[j] = sum + b1[j];
        }
        
        // 统计激活的隐藏神经元
        int activeCount = 0;
        boolean[] active = new boolean[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            active[j] = z1Mean[j] > 0;
            if (active[j]) activeCount++;
        }
        System.out.println("激活的隐藏神经元: " + activeCount + " / " + hiddenSize);
        
        // 计算等效线性系数：k_i = Σ_j (W2[j] * W1[i][j]) * active[j]
        // 这是在 ReLU 未饱和区域的线性近似
        double[] equivCoeff = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++) {
                if (active[j]) {
                    sum += W2[j][0] * W1[i][j];
                }
            }
            equivCoeff[i] = sum;
        }
        
        // 计算等效偏置：b_eff = Σ_j (W2[j] * b1[j]) * active[j] + b2[0]
        double equivBias = 0;
        for (int j = 0; j < hiddenSize; j++) {
            if (active[j]) {
                equivBias += W2[j][0] * b1[j];
            }
        }
        equivBias += b2[0];
        
        // 打印等效线性公式（原始尺度）
        // 需要将标准化空间的系数转换到原始尺度
        System.out.println("\n标准化空间:");
        System.out.print("  ŷ_norm = ");
        boolean first = true;
        for (int i = 0; i < inputSize; i++) {
            if (Math.abs(equivCoeff[i]) < 1e-6) continue;
            if (!first && equivCoeff[i] > 0) System.out.print(" + ");
            if (!first && equivCoeff[i] < 0) System.out.print(" - ");
            if (first && equivCoeff[i] < 0) System.out.print("-");
            System.out.printf("%.4f·x%d_norm", Math.abs(equivCoeff[i]), i + 1);
            first = false;
        }
        System.out.printf(" + %.4f\n", equivBias);
        
        // 原始空间：y = ŷ_norm * targetStd + targetMean
        // x_i_norm = (x_i - mean_i) / std_i
        // 展开后：y = Σ (coeff_i * x_i / std_i) + (b_eff - Σ coeff_i * mean_i / std_i) * targetStd + targetMean
        System.out.println("\n原始空间 (类似最小二乘法):");
        System.out.print("  y = ");
        first = true;
        for (int i = 0; i < inputSize; i++) {
            double origCoeff = equivCoeff[i] * targetStd / featureStd[i];
            if (Math.abs(origCoeff) < 1e-6) continue;
            if (!first && origCoeff > 0) System.out.print(" + ");
            if (!first && origCoeff < 0) System.out.print(" - ");
            if (first && origCoeff < 0) System.out.print("-");
            System.out.printf("%.4f·x%d", Math.abs(origCoeff), i + 1);
            first = false;
        }
        // 计算原始空间的常数项
        double origConst = equivBias * targetStd + targetMean;
        for (int i = 0; i < inputSize; i++) {
            double origCoeff = equivCoeff[i] * targetStd / featureStd[i];
            origConst -= origCoeff * featureMean[i];
        }
        System.out.printf(" + %.4f\n", origConst);
        
        System.out.println("\n说明: 以上等效公式为 ReLU 网络在样本均值附近的线性近似，");
        System.out.println("      实际网络为非线性模型，不同输入区域的等效公式会不同。");
        System.out.println("      所有计算在标准化空间完成，预测时会自动反标准化到原始尺度。");
    }
    
    // ==================== 评估 ====================
    public void evaluate(double[][] X, double[] y) {
        double[] y_pred = predict(X);
        int m = X.length;
        
        double mse = 0;
        double mae = 0;
        for (int i = 0; i < m; i++) {
            double diff = y_pred[i] - y[i];
            mse += diff * diff;
            mae += Math.abs(diff);
        }
        mse /= m;
        mae /= m;
        
        System.out.println("\n========== 评估结果 ==========");
        System.out.printf("MSE (均方误差): %.4f\n", mse);
        System.out.printf("MAE (平均绝对误差): %.4f\n", mae);
        
        System.out.println("\n预测值 vs 真实值:");
        for (int i = 0; i < Math.min(10, m); i++) {
            System.out.printf(" 样本%d: 预测=%.2f, 真实=%.2f, 误差=%.2f\n", 
                i + 1, y_pred[i], y[i], y_pred[i] - y[i]);
        }
    }
    
    // ==================== 主函数（测试） ====================
    public static void main(String[] args) {
        // ===== 全部数据 =====
        // 特征: [面积, 房龄, 房间数]
        double[][] X_all = {
            {50, 2, 2}, {80, 5, 3}, {100, 3, 4}, {120, 1, 5},
            {70, 8, 3}, {90, 4, 4}, {110, 2, 5}, {60, 6, 2},
            {130, 1, 6}, {85, 3, 3}, {95, 2, 4}, {75, 7, 3},
            {105, 3, 5}, {115, 1, 5}, {65, 5, 2}, {55, 4, 2},
            {72, 6, 3}, {88, 3, 3}, {102, 2, 4}, {118, 1, 5},
            {68, 7, 2}, {78, 5, 3}, {92, 4, 4}, {108, 2, 5},
            {125, 1, 6}, {135, 1, 6}, {82, 4, 3}, {97, 3, 4},
            {112, 2, 5}, {62, 6, 2},
            {48, 3, 2}, {58, 5, 2}, {67, 4, 2}, {73, 6, 3},
            {76, 5, 3}, {83, 3, 3}, {86, 4, 3}, {93, 2, 4},
            {98, 3, 4}, {103, 4, 4}, {107, 2, 5}, {113, 1, 5},
            {117, 2, 5}, {122, 1, 5}, {128, 1, 6}, {132, 1, 6},
            {45, 8, 2}, {52, 7, 2}, {89, 2, 4}, {140, 1, 6}
        };
        
        double[] y_all = {160, 250, 310, 380, 210, 280, 360, 190,
                          410, 270, 290, 220, 340, 370, 180, 170,
                          215, 275, 315, 375, 200, 240, 285, 350,
                          400, 420, 260, 300, 355, 195,
                          155, 175, 205, 218, 238, 272, 273, 310,
                          305, 320, 345, 368, 372, 385, 405, 418,
                          150, 165, 282, 430};
        
        // ===== 划分训练集和测试集（80%训练，20%测试）======
        int total = X_all.length;
        int trainSize = (int)(total * 0.8);  // 12个训练
        int testSize = total - trainSize;     // 3个测试
        
        // 打乱索引
        int[] indices = new int[total];
        for (int i = 0; i < total; i++) indices[i] = i;
        Random shuffleRand = new Random(42);
        for (int i = total - 1; i > 0; i--) {
            int j = shuffleRand.nextInt(i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
        
        double[][] X_train = new double[trainSize][];
        double[] y_train = new double[trainSize];
        double[][] X_test = new double[testSize][];
        double[] y_test = new double[testSize];
        
        for (int i = 0; i < trainSize; i++) {
            X_train[i] = X_all[indices[i]];
            y_train[i] = y_all[indices[i]];
        }
        for (int i = 0; i < testSize; i++) {
            X_test[i] = X_all[indices[trainSize + i]];
            y_test[i] = y_all[indices[trainSize + i]];
        }
        
        System.out.println("========== 数据集划分 ==========");
        System.out.println("总样本: " + total + ", 训练集: " + trainSize + ", 测试集: " + testSize);
        System.out.print("训练集索引: ");
        for (int i = 0; i < trainSize; i++) System.out.print(indices[i] + " ");
        System.out.print("\n测试集索引: ");
        for (int i = 0; i < testSize; i++) System.out.print(indices[trainSize + i] + " ");
        System.out.println("\n");
        
        // ===== 创建模型（小网络 + 强正则化 + 早停） =====
        TwoLayerNeuralNetworkReLU model = new TwoLayerNeuralNetworkReLU(3, 4, 0.005, 5000, 0.1);
        
        // ===== 训练 =====
        model.train(X_train, y_train);
        
        // ===== 打印权重和公式 =====
        model.printWeightsAndFormula();
        
        // ===== 训练集评估 =====
                                                                System.out.println("\n==================== 训练集评估 ====================");
        model.evaluate(X_train, y_train);
        
        // ===== 测试集评估 =====
        System.out.println("\n==================== 测试集评估 ====================");
        model.evaluate(X_test, y_test);
        
        // ===== 过拟合判断 =====
        double trainMSE = calcMSE(model, X_train, y_train);
        double testMSE = calcMSE(model, X_test, y_test);
        System.out.println("\n========== 过拟合分析 ==========");
        System.out.printf("训练集 MSE: %.4f\n", trainMSE);
        System.out.printf("测试集 MSE: %.4f\n", testMSE);
        System.out.printf("测试/训练 MSE 比值: %.2f\n", testMSE / trainMSE);
        if (testMSE / trainMSE > 3.0) {
            System.out.println("结论: 存在明显过拟合！测试集误差远大于训练集");
        } else if (testMSE / trainMSE > 1.5) {
            System.out.println("结论: 存在轻微过拟合");
        } else {
            System.out.println("结论: 未发现明显过拟合");
        }
        
        // ===== 单样本预测 =====
        double[] newHouse = {95, 2, 4};
        double price = model.predict(newHouse);
        System.out.printf("\n新房子 [面积=%.0f, 房龄=%.0f, 房间数=%.0f] 预测房价: %.2f 万\n", 
            newHouse[0], newHouse[1], newHouse[2], price);
    }
    
    private static double calcMSE(TwoLayerNeuralNetworkReLU model, double[][] X, double[] y) {
        double[] pred = model.predict(X);
        double mse = 0;
        for (int i = 0; i < X.length; i++) {
            double diff = pred[i] - y[i];
            mse += diff * diff;
        }
        return mse / X.length;
    }
}