package com.mv.demo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * HTML 报告生成器：生成独立的可视化对比报告页面。
 * <p>
 * 使用 Chart.js (CDN) 绘制 4 张图表：
 * <ol>
 *   <li>实际值 vs OLS 预测值 散点图</li>
 *   <li>实际值 vs NN 预测值 散点图</li>
 *   <li>两种模型的误差分布柱状对比图</li>
 *   <li>前 200 个样本的预测曲线对比图</li>
 * </ol>
 * </p>
 */
public class HtmlReporter {

    /**
     * 生成 HTML 报告文件。
     *
     * @param filePath      输出文件路径
     * @param actual        实际值数组
     * @param olsPredicted  OLS 预测值数组
     * @param nnPredicted   NN 预测值数组
     * @param olsMetrics    OLS 评估指标
     * @param nnMetrics     NN 评估指标
     * @param olsFormula    OLS 回归公式
     * @param nnParamCount  NN 参数数量
     * @param trainSize     训练集大小
     * @param testSize      测试集大小
     */
    public void generate(String filePath,
                         double[] actual,
                         double[] olsPredicted,
                         double[] nnPredicted,
                         ModelEvaluator.Metrics olsMetrics,
                         ModelEvaluator.Metrics nnMetrics,
                         String olsFormula,
                         int nnParamCount,
                         int trainSize,
                         int testSize) throws IOException {

        int n = actual.length;
        int plotN = Math.min(200, n);

        // 计算误差分布（柱状图数据）
        int numBins = 20;
        double[] olsErrors = new double[n];
        double[] nnErrors = new double[n];
        for (int i = 0; i < n; i++) {
            olsErrors[i] = actual[i] - olsPredicted[i];
            nnErrors[i] = actual[i] - nnPredicted[i];
        }

        double minErr = Double.MAX_VALUE, maxErr = -Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            minErr = Math.min(minErr, Math.min(olsErrors[i], nnErrors[i]));
            maxErr = Math.max(maxErr, Math.max(olsErrors[i], nnErrors[i]));
        }
        double binWidth = (maxErr - minErr) / numBins;
        int[] olsBins = new int[numBins];
        int[] nnBins = new int[numBins];
        String[] binLabels = new String[numBins];
        for (int b = 0; b < numBins; b++) {
            double lo = minErr + b * binWidth;
            double hi = lo + binWidth;
            binLabels[b] = String.format(Locale.US, "%.0f~%.0f", lo, hi);
            for (int i = 0; i < n; i++) {
                if (olsErrors[i] >= lo && olsErrors[i] < hi) olsBins[b]++;
                if (nnErrors[i] >= lo && nnErrors[i] < hi) nnBins[b]++;
            }
        }

        // 构建 JSON 数组
        StringBuilder actualJson = new StringBuilder("[");
        StringBuilder olsJson = new StringBuilder("[");
        StringBuilder nnJson = new StringBuilder("[");
        for (int i = 0; i < n; i++) {
            if (i > 0) { actualJson.append(","); olsJson.append(","); nnJson.append(","); }
            actualJson.append(String.format(Locale.US, "%.2f", actual[i]));
            olsJson.append(String.format(Locale.US, "%.2f", olsPredicted[i]));
            nnJson.append(String.format(Locale.US, "%.2f", nnPredicted[i]));
        }
        actualJson.append("]");
        olsJson.append("]");
        nnJson.append("]");

        // 前 plotN 个样本的 JSON
        StringBuilder plotActual = new StringBuilder("[");
        StringBuilder plotOls = new StringBuilder("[");
        StringBuilder plotNn = new StringBuilder("[");
        StringBuilder plotLabels = new StringBuilder("[");
        for (int i = 0; i < plotN; i++) {
            if (i > 0) {
                plotActual.append(",");
                plotOls.append(",");
                plotNn.append(",");
                plotLabels.append(",");
            }
            plotActual.append(String.format(Locale.US, "%.2f", actual[i]));
            plotOls.append(String.format(Locale.US, "%.2f", olsPredicted[i]));
            plotNn.append(String.format(Locale.US, "%.2f", nnPredicted[i]));
            plotLabels.append(i);
        }
        plotActual.append("]");
        plotOls.append("]");
        plotNn.append("]");
        plotLabels.append("]");

        // 误差分布 JSON
        StringBuilder olsBinsJson = new StringBuilder("[");
        StringBuilder nnBinsJson = new StringBuilder("[");
        StringBuilder binLabelsJson = new StringBuilder("[");
        for (int b = 0; b < numBins; b++) {
            if (b > 0) {
                olsBinsJson.append(",");
                nnBinsJson.append(",");
                binLabelsJson.append(",");
            }
            olsBinsJson.append(olsBins[b]);
            nnBinsJson.append(nnBins[b]);
            binLabelsJson.append("\"").append(binLabels[b]).append("\"");
        }
        olsBinsJson.append("]");
        nnBinsJson.append("]");
        binLabelsJson.append("]");

        // 精度提升百分比
        double r2Improve = ((nnMetrics.r2 - olsMetrics.r2) / olsMetrics.r2) * 100.0;
        double cvrmseReduce = ((olsMetrics.cvrmse - nnMetrics.cvrmse) / olsMetrics.cvrmse) * 100.0;

        String html = "<!DOCTYPE html>\n"
                + "<html lang=\"zh-CN\">\n"
                + "<head>\n"
                + "<meta charset=\"UTF-8\">\n"
                + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
                + "<title>M&amp;V 能耗预测对比报告</title>\n"
                + "<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js\"></script>\n"
                + "<style>\n"
                + "  * { margin: 0; padding: 0; box-sizing: border-box; }\n"
                + "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n"
                + "         background: #f0f2f5; color: #333; line-height: 1.6; }\n"
                + "  .container { max-width: 1200px; margin: 0 auto; padding: 20px; }\n"
                + "  h1 { text-align: center; color: #1a73e8; margin: 30px 0 10px; font-size: 28px; }\n"
                + "  .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }\n"
                + "  .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);\n"
                + "          padding: 24px; margin-bottom: 24px; }\n"
                + "  .card h2 { color: #1a73e8; font-size: 18px; margin-bottom: 16px;\n"
                + "             border-bottom: 2px solid #e8eaed; padding-bottom: 8px; }\n"
                + "  .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }\n"
                + "  @media (max-width: 768px) { .chart-row { grid-template-columns: 1fr; } }\n"
                + "  .chart-box { background: #fff; border-radius: 12px;\n"
                + "               box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 20px; }\n"
                + "  .chart-box h3 { font-size: 15px; color: #555; margin-bottom: 12px; text-align: center; }\n"
                + "  table { width: 100%; border-collapse: collapse; margin-top: 12px; }\n"
                + "  th, td { padding: 10px 14px; text-align: center; border: 1px solid #e0e0e0; }\n"
                + "  th { background: #1a73e8; color: #fff; font-weight: 600; }\n"
                + "  tr:nth-child(even) { background: #f8f9fa; }\n"
                + "  .highlight { font-weight: bold; color: #0d904f; }\n"
                + "  .tag-ols { display: inline-block; padding: 2px 10px; border-radius: 12px;\n"
                + "             background: #e3f2fd; color: #1565c0; font-size: 13px; font-weight: 600; }\n"
                + "  .tag-nn { display: inline-block; padding: 2px 10px; border-radius: 12px;\n"
                + "            background: #e8f5e9; color: #2e7d32; font-size: 13px; font-weight: 600; }\n"
                + "  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n"
                + "                  gap: 16px; margin-top: 16px; }\n"
                + "  .metric-card { background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }\n"
                + "  .metric-card .label { font-size: 13px; color: #888; margin-bottom: 4px; }\n"
                + "  .metric-card .value { font-size: 24px; font-weight: 700; color: #1a73e8; }\n"
                + "  .formula { background: #f1f3f4; border-radius: 8px; padding: 14px 18px;\n"
                + "             font-family: 'Courier New', monospace; font-size: 13px; color: #333;\n"
                + "             overflow-x: auto; margin-top: 8px; }\n"
                + "  footer { text-align: center; color: #999; font-size: 12px; margin-top: 30px; padding: 20px 0; }\n"
                + "</style>\n"
                + "</head>\n"
                + "<body>\n"
                + "<div class=\"container\">\n"
                + "  <h1>📊 M&amp;V 能耗预测对比报告</h1>\n"
                + "  <p class=\"subtitle\">OLS 回归 vs 神经网络 (MLP) — 基于模拟 M&amp;V 场景数据</p>\n"

                // 数据概览
                + "  <div class=\"card\">\n"
                + "    <h2>📋 数据概览</h2>\n"
                + "    <div class=\"summary-grid\">\n"
                + "      <div class=\"metric-card\"><div class=\"label\">总数据量</div><div class=\"value\">" + (trainSize + testSize) + "</div></div>\n"
                + "      <div class=\"metric-card\"><div class=\"label\">训练集</div><div class=\"value\">" + trainSize + "</div></div>\n"
                + "      <div class=\"metric-card\"><div class=\"label\">测试集</div><div class=\"value\">" + testSize + "</div></div>\n"
                + "      <div class=\"metric-card\"><div class=\"label\">特征维度</div><div class=\"value\">3</div></div>\n"
                + "    </div>\n"
                + "  </div>\n"

                // 模型信息
                + "  <div class=\"card\">\n"
                + "    <h2>🔧 模型配置</h2>\n"
                + "    <p><span class=\"tag-ols\">OLS</span> " + olsFormula + "</p>\n"
                + "    <p style=\"margin-top:8px\"><span class=\"tag-nn\">MLP</span> 结构: 3 → 64(ReLU) → 32(ReLU) → 1 &nbsp;|&nbsp; 参数数量: " + String.format("%,d", nnParamCount) + "</p>\n"
                + "  </div>\n"

                // 散点图
                + "  <div class=\"chart-row\">\n"
                + "    <div class=\"chart-box\">\n"
                + "      <h3>实际值 vs OLS 预测值</h3>\n"
                + "      <canvas id=\"scatterOLS\"></canvas>\n"
                + "    </div>\n"
                + "    <div class=\"chart-box\">\n"
                + "      <h3>实际值 vs NN 预测值</h3>\n"
                + "      <canvas id=\"scatterNN\"></canvas>\n"
                + "    </div>\n"
                + "  </div>\n"

                // 误差分布 + 预测曲线
                + "  <div class=\"chart-row\">\n"
                + "    <div class=\"chart-box\">\n"
                + "      <h3>误差分布对比</h3>\n"
                + "      <canvas id=\"errorDist\"></canvas>\n"
                + "    </div>\n"
                + "    <div class=\"chart-box\">\n"
                + "      <h3>前 " + plotN + " 个样本预测曲线对比</h3>\n"
                + "      <canvas id=\"predCurve\"></canvas>\n"
                + "    </div>\n"
                + "  </div>\n"

                // 对比总结表格
                + "  <div class=\"card\">\n"
                + "    <h2>📈 评估指标对比</h2>\n"
                + "    <table>\n"
                + "      <tr><th>指标</th><th><span class=\"tag-ols\">OLS</span></th><th><span class=\"tag-nn\">MLP</span></th><th>优势</th></tr>\n"
                + "      <tr><td>MSE</td><td>" + String.format(Locale.US, "%.4f", olsMetrics.mse) + "</td>"
                + "<td>" + String.format(Locale.US, "%.4f", nnMetrics.mse) + "</td>"
                + "<td class=\"highlight\">" + (nnMetrics.mse < olsMetrics.mse ? "MLP ✓" : "OLS ✓") + "</td></tr>\n"
                + "      <tr><td>MAE</td><td>" + String.format(Locale.US, "%.4f", olsMetrics.mae) + "</td>"
                + "<td>" + String.format(Locale.US, "%.4f", nnMetrics.mae) + "</td>"
                + "<td class=\"highlight\">" + (nnMetrics.mae < olsMetrics.mae ? "MLP ✓" : "OLS ✓") + "</td></tr>\n"
                + "      <tr><td>R²</td><td>" + String.format(Locale.US, "%.4f", olsMetrics.r2) + "</td>"
                + "<td>" + String.format(Locale.US, "%.4f", nnMetrics.r2) + "</td>"
                + "<td class=\"highlight\">" + (nnMetrics.r2 > olsMetrics.r2 ? "MLP ✓" : "OLS ✓") + "</td></tr>\n"
                + "      <tr><td>CVRMSE (%)</td><td>" + String.format(Locale.US, "%.2f", olsMetrics.cvrmse) + "</td>"
                + "<td>" + String.format(Locale.US, "%.2f", nnMetrics.cvrmse) + "</td>"
                + "<td class=\"highlight\">" + (nnMetrics.cvrmse < olsMetrics.cvrmse ? "MLP ✓" : "OLS ✓") + "</td></tr>\n"
                + "      <tr><td>NMBE (%)</td><td>" + String.format(Locale.US, "%.2f", olsMetrics.nmbe) + "</td>"
                + "<td>" + String.format(Locale.US, "%.2f", nnMetrics.nmbe) + "</td>"
                + "<td class=\"highlight\">" + (Math.abs(nnMetrics.nmbe) < Math.abs(olsMetrics.nmbe) ? "MLP ✓" : "OLS ✓") + "</td></tr>\n"
                + "    </table>\n"
                + "    <div class=\"summary-grid\" style=\"margin-top:20px\">\n"
                + "      <div class=\"metric-card\"><div class=\"label\">R² 提升</div>"
                + "<div class=\"value\" style=\"color:#0d904f\">+" + String.format(Locale.US, "%.1f", r2Improve) + "%</div></div>\n"
                + "      <div class=\"metric-card\"><div class=\"label\">CVRMSE 降低</div>"
                + "<div class=\"value\" style=\"color:#0d904f\">-" + String.format(Locale.US, "%.1f", cvrmseReduce) + "%</div></div>\n"
                + "    </div>\n"
                + "  </div>\n"

                + "  <footer>M&amp;V Energy Prediction Comparison Report &bull; Generated by Java MLP vs OLS Demo</footer>\n"
                + "</div>\n"

                // Chart.js 脚本
                + "<script>\n"
                + "const actualData = " + actualJson + ";\n"
                + "const olsData = " + olsJson + ";\n"
                + "const nnData = " + nnJson + ";\n"
                + "\n"
                // 散点图辅助数据（降采样以避免浏览器卡顿）
                + "const scatterN = Math.min(800, actualData.length);\n"
                + "const step = Math.max(1, Math.floor(actualData.length / scatterN));\n"
                + "let sActual=[], sOls=[], sNn=[];\n"
                + "for(let i=0;i<actualData.length;i+=step){sActual.push(actualData[i]);sOls.push(olsData[i]);sNn.push(nnData[i]);}\n"
                + "\n"
                // 1. OLS 散点图
                + "new Chart(document.getElementById('scatterOLS'),{\n"
                + "  type:'scatter',\n"
                + "  data:{datasets:[\n"
                + "    {data:sActual.map((v,i)=>({x:v,y:sOls[i]})),backgroundColor:'rgba(21,101,192,0.4)',pointRadius:2,label:'OLS 预测'},\n"
                + "    {data:[{x:Math.min(...sActual),y:Math.min(...sActual)},{x:Math.max(...sActual),y:Math.max(...sActual)}],"
                + "     type:'line',borderColor:'#e53935',borderWidth:1.5,pointRadius:0,label:'理想线',borderDash:[5,5]}\n"
                + "  ]},options:{scales:{x:{title:{display:true,text:'实际值 (kWh)'}},y:{title:{display:true,text:'OLS 预测值 (kWh)'}}},plugins:{legend:{display:true,position:'bottom'}}}\n"
                + "});\n"
                + "\n"
                // 2. NN 散点图
                + "new Chart(document.getElementById('scatterNN'),{\n"
                + "  type:'scatter',\n"
                + "  data:{datasets:[\n"
                + "    {data:sActual.map((v,i)=>({x:v,y:sNn[i]})),backgroundColor:'rgba(46,125,50,0.4)',pointRadius:2,label:'NN 预测'},\n"
                + "    {data:[{x:Math.min(...sActual),y:Math.min(...sActual)},{x:Math.max(...sActual),y:Math.max(...sActual)}],"
                + "     type:'line',borderColor:'#e53935',borderWidth:1.5,pointRadius:0,label:'理想线',borderDash:[5,5]}\n"
                + "  ]},options:{scales:{x:{title:{display:true,text:'实际值 (kWh)'}},y:{title:{display:true,text:'NN 预测值 (kWh)'}}},plugins:{legend:{display:true,position:'bottom'}}}\n"
                + "});\n"
                + "\n"
                // 3. 误差分布柱状图
                + "new Chart(document.getElementById('errorDist'),{\n"
                + "  type:'bar',\n"
                + "  data:{labels:" + binLabelsJson + ",datasets:[\n"
                + "    {label:'OLS 误差',data:" + olsBinsJson + ",backgroundColor:'rgba(21,101,192,0.6)'},\n"
                + "    {label:'NN 误差',data:" + nnBinsJson + ",backgroundColor:'rgba(46,125,50,0.6)'}\n"
                + "  ]},options:{scales:{x:{title:{display:true,text:'误差区间 (kWh)'}},y:{title:{display:true,text:'频次'}}},plugins:{legend:{position:'bottom'}}}\n"
                + "});\n"
                + "\n"
                // 4. 预测曲线对比
                + "new Chart(document.getElementById('predCurve'),{\n"
                + "  type:'line',\n"
                + "  data:{labels:" + plotLabels + ",datasets:[\n"
                + "    {label:'实际值',data:" + plotActual + ",borderColor:'#333',borderWidth:1.5,pointRadius:0,fill:false},\n"
                + "    {label:'OLS 预测',data:" + plotOls + ",borderColor:'#1565c0',borderWidth:1.2,pointRadius:0,fill:false},\n"
                + "    {label:'NN 预测',data:" + plotNn + ",borderColor:'#2e7d32',borderWidth:1.2,pointRadius:0,fill:false}\n"
                + "  ]},options:{scales:{x:{title:{display:true,text:'样本序号'},ticks:{maxTicksLimit:20}},y:{title:{display:true,text:'能耗 (kWh)'}}},plugins:{legend:{position:'bottom'}},elements:{line:{tension:0.1}}}\n"
                + "});\n"
                + "</script>\n"
                + "</body>\n"
                + "</html>";

        Files.write(Paths.get(filePath), html.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }
}
