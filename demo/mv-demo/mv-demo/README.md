# M&V 能耗预测对比：OLS vs 神经网络

纯 Java Maven 项目，对比普通最小二乘 (OLS) 回归与多层感知器 (MLP) 神经网络在 M&V（测量与验证）能耗预测场景下的效果。

## 项目结构

```
mv-demo/
├── pom.xml                          # Maven 配置
├── mv_comparison_report.html        # 生成的 HTML 可视化报告
├── README.md
└── src/main/java/com/mv/demo/
    ├── Main.java                    # 主入口，串联整个流程
    ├── DataGenerator.java           # 模拟数据生成（12个月逐时数据）
    ├── OLSModel.java                # OLS 回归（Apache Commons Math）
    ├── NNModel.java                 # MLP 神经网络（DJL + PyTorch）
    ├── ModelEvaluator.java          # 评估指标计算（MSE/MAE/R²/CVRMSE/NMBE）
    └── HtmlReporter.java            # HTML 报告生成器（Chart.js）
```

## 环境要求

- JDK 11+
- Maven 3.6+

## 快速开始

```bash
# 编译
mvn clean compile

# 运行
mvn exec:java -Dexec.mainClass="com.mv.demo.Main"
```

或者打包后运行：

```bash
mvn clean package
java -cp "target/classes:$(mvn dependency:build-classpath -q -DincludeScope=runtime -Dmdep.outputFile=/dev/stdout)" com.mv.demo.Main
```

## 数据说明

生成 12 个月的逐时模拟数据（约 8760 条），模拟真实 M&V 场景：

- **temperature**: 温度 (°C)，含年度正弦趋势 + 日变化
- **humidity**: 湿度 (%)，与温度负相关
- **production**: 生产量，工作时段较高
- **energy_consumption**: 能耗目标值，由以下非线性公式生成：
  ```
  energy = 0.3 × temp² + 2.0 × humidity + 1.5 × production
         + 0.02 × temp × production + noise(σ=15)
  ```

## 模型配置

| 模型 | 框架 | 结构 |
|------|------|------|
| OLS | Apache Commons Math | 线性回归 |
| MLP | DJL (PyTorch) | 3 → 64(ReLU) → 32(ReLU) → 1 |

## 输出

- **控制台**: 完整的指标对比
- **HTML 报告** (`mv_comparison_report.html`): 包含 4 张 Chart.js 图表 + 对比总结表格

## 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Apache Commons Math | 3.6.1 | OLS 回归 |
| DJL API | 0.26.0 | 深度学习框架 |
| DJL PyTorch Engine | 0.26.0 | 后端引擎 |
