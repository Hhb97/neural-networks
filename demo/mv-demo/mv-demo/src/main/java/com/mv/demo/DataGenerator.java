package com.mv.demo;

import java.time.LocalDateTime;
import java.time.Month;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 数据生成器：模拟 M&V（测量与验证）场景下的能耗数据。
 * <p>
 * 生成 12 个月的逐时数据（约 8760 条），包含温度、湿度、生产量三个特征变量，
 * 以及由非线性关系决定的能耗目标值。
 * </p>
 *
 * <p>真实数据生成公式：</p>
 * <pre>
 * energy = 0.3 * temperature² + 2.0 * humidity + 1.5 * production
 *        + 0.02 * temperature * production + noise(stddev=15)
 * </pre>
 */
public class DataGenerator {

    /** 单条数据记录 */
    public static class DataPoint {
        public final LocalDateTime timestamp;
        public final double temperature;
        public final double humidity;
        public final double production;
        public final double energyConsumption;

        public DataPoint(LocalDateTime timestamp, double temperature,
                         double humidity, double production, double energyConsumption) {
            this.timestamp = timestamp;
            this.temperature = temperature;
            this.humidity = humidity;
            this.production = production;
            this.energyConsumption = energyConsumption;
        }
    }

    private static final long SEED = 42L;

    /**
     * 生成 12 个月的逐时模拟数据。
     *
     * @return 数据列表，共约 8760 条
     */
    public List<DataPoint> generate() {
        Random rng = new Random(SEED);
        List<DataPoint> data = new ArrayList<>();

        LocalDateTime start = LocalDateTime.of(2024, Month.JANUARY, 1, 0, 0);
        LocalDateTime end = LocalDateTime.of(2025, Month.JANUARY, 1, 0, 0);

        LocalDateTime current = start;
        while (current.isBefore(end)) {
            // 温度：基于天的正弦变化 + 年度趋势
            int dayOfYear = current.getDayOfYear();
            int hour = current.getHour();
            double tempBase = 15.0 + 25.0 * Math.sin(2.0 * Math.PI * (dayOfYear - 80) / 365.0);
            double tempDiurnal = 5.0 * Math.sin(2.0 * Math.PI * (hour - 6) / 24.0);
            double temperature = tempBase + tempDiurnal + rng.nextGaussian() * 2.0;

            // 湿度：与温度负相关
            double humidity = 60.0 - 0.3 * temperature + rng.nextGaussian() * 5.0;
            humidity = Math.max(30.0, Math.min(90.0, humidity));

            // 生产量：工作时段较高
            double prodBase = (hour >= 8 && hour < 20) ? 150.0 : 80.0;
            double production = prodBase + rng.nextGaussian() * 20.0;
            production = Math.max(50.0, Math.min(200.0, production));

            // 能耗：非线性关系 + 噪声
            double energy = 0.3 * temperature * temperature
                    + 2.0 * humidity
                    + 1.5 * production
                    + 0.02 * temperature * production
                    + rng.nextGaussian() * 15.0;

            data.add(new DataPoint(current, temperature, humidity, production, energy));
            current = current.plusHours(1);
        }

        return data;
    }
}
