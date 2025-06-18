# Professional Funnel Analytics Platform

An enterprise-grade funnel analytics platform built with Streamlit, featuring real-time calculations, multiple data sources, and advanced funnel analysis capabilities.

## 🚀 Features

### Core Analytics Engine
- **Real Funnel Calculation**: Process actual event data with user IDs, event names, and timestamps
- **Multiple Counting Methods**:
  - `unique_users`: Count unique users progressing through each step
  - `event_totals`: Count total events at each step
  - `unique_pairs`: Step-to-step conversion analysis
- **Flexible Conversion Windows**: Configure time windows (hours/days/weeks) for funnel completion
- **Re-entry Mode Logic**: Handle users who restart the funnel process

### Data Sources
- **File Upload**: Support for CSV and Parquet files
- **ClickHouse Integration**: Enterprise OLAP database connectivity
- **Sample Data**: Built-in demonstration datasets

### Advanced Visualizations
- **Professional Funnel Charts**: Interactive Plotly visualizations
- **Sankey Flow Diagrams**: User journey flow analysis
- **Real-time Metrics**: Dynamic conversion rate calculations

## 📋 Data Requirements

### Required Columns
Your event data must contain these columns:
- `user_id`: Unique identifier for each user
- `event_name`: Name of the event/action performed
- `timestamp`: When the event occurred (ISO format recommended)

### Optional Columns
- `event_properties`: JSON string with additional event metadata for segmentation

### Sample Data Format
```csv
user_id,event_name,timestamp,event_properties
user_00001,User Sign-Up,2024-01-01 10:00:00,"{""platform"": ""mobile""}"
user_00001,Verify Email,2024-01-01 10:15:00,"{""platform"": ""mobile""}"
user_00001,First Login,2024-01-01 11:30:00,"{""platform"": ""mobile""}"
```

## 🔧 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## 📊 Usage

### 1. Data Source Configuration
Choose your data source from the sidebar:
- **Sample Data**: Use built-in demonstration data
- **Upload File**: Upload CSV or Parquet files
- **ClickHouse**: Connect to your enterprise database

### 2. ClickHouse Integration
For enterprise deployments, configure ClickHouse connection:
```sql
-- Sample query structure
SELECT 
    user_id,
    event_name,
    timestamp,
    event_properties
FROM events 
WHERE timestamp >= '2024-01-01' 
ORDER BY user_id, timestamp
```

### 3. Funnel Configuration
Configure analysis parameters:
- **Conversion Window**: Time limit for users to complete the funnel
- **Counting Method**: How to count conversions
- **Re-entry Mode**: How to handle funnel restarts

### 4. Building Funnels
1. Select events from your data
2. Add them to the funnel in sequence
3. Run the analysis
4. View results in multiple visualization formats

## 🏗️ Architecture

### Modular Design
- `DataSourceManager`: Handles multiple data sources and validation
- `FunnelCalculator`: Core calculation engine with multiple algorithms
- `FunnelVisualizer`: Professional visualization components
- `FunnelConfig` & `FunnelResults`: Type-safe data structures

### Calculation Methods

#### Unique Users Method
Tracks unique users progressing through each funnel step within the conversion window.

#### Event Totals Method
Counts total events at each step, useful for understanding event volume.

#### Unique Pairs Method
Analyzes step-to-step conversions, ideal for identifying specific drop-off points.

### Re-entry Modes

#### First Only
Users are only counted on their first attempt through the funnel.

#### Optimized Re-entry
Users can be counted multiple times if they restart the funnel process.

## 🎯 Key Metrics

The platform calculates and displays:
- **Overall Conversion Rate**: End-to-end funnel completion
- **Step-by-Step Conversion**: Individual step performance
- **Drop-off Analysis**: Where users leave the funnel
- **User Journey Flow**: Visual representation of user paths

## 🔐 Enterprise Features

- **Scalable Data Processing**: Handles large datasets efficiently
- **Database Integration**: Direct connection to ClickHouse and other OLAP databases
- **Real-time Calculations**: Dynamic funnel metrics without pre-aggregation
- **Professional Visualizations**: Export-ready charts and diagrams
- **Flexible Configuration**: Customizable analysis parameters

## 📈 Performance

- **Optimized Algorithms**: Efficient funnel calculation for large datasets
- **Caching**: Built-in Streamlit caching for improved performance
- **Modular Architecture**: Easy to extend and maintain

## 🤝 Contributing

This platform is designed for enterprise use cases. For customization:

1. Extend `DataSourceManager` for new data sources
2. Add calculation methods to `FunnelCalculator`
3. Create new visualizations in `FunnelVisualizer`
4. Define custom configuration options in `FunnelConfig`

### Development & Testing

For developers working on the platform:
- **Testing Documentation**: See [`tests/README.md`](tests/README.md) for comprehensive testing architecture
- **Test Data**: See [`test_data/README.md`](test_data/README.md) for test data organization
- **Run Tests**: Use `python run_tests.py` for professional test execution

## 📄 License

Enterprise-grade funnel analytics platform for professional use. 