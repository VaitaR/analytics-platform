# Python Version Compatibility

## Supported Python Versions

This project supports the following Python versions:
- **Python 3.9** - Minimum supported version
- **Python 3.10** - Fully supported
- **Python 3.11** - Fully supported  
- **Python 3.12** - Fully supported

## Python 3.8 Support Removed

### Reason for Removal

Python 3.8 support was removed due to dependency compatibility issues:

1. **scipy>=1.11.0** - Required for statistical functions, but not available for Python 3.8 (max version: 1.10.1)
2. **pandas>=2.0.0** - Modern pandas versions have limited Python 3.8 support
3. **numpy>=1.24.0** - Performance optimizations require newer numpy versions
4. **streamlit>=1.28.0** - Latest Streamlit features require Python 3.9+

### Python 3.8 End of Life

Python 3.8 reached end-of-life status in **October 2024**, meaning:
- No more security updates
- Many popular libraries are dropping support
- Recommended to migrate to Python 3.9+

## GitHub Actions Updates

### Changes Made

1. **Python Version Matrix**: Updated from `[3.8, 3.9, "3.10", "3.11"]` to `[3.9, "3.10", "3.11", "3.12"]`

2. **Actions Versions Updated**:
   - `actions/setup-python@v4` → `actions/setup-python@v5`
   - `actions/cache@v3` → `actions/cache@v4`
   - `codecov/codecov-action@v3` → `codecov/codecov-action@v4`
   - `actions/upload-artifact@v3` → `actions/upload-artifact@v4`

3. **Standardized Python Version**: All specialized jobs now use Python 3.11 for consistency

### Test Matrix

The updated CI pipeline runs tests on:
- **Main test job**: Python 3.9, 3.10, 3.11, 3.12
- **Category tests**: Python 3.11 (standardized)
- **Performance tests**: Python 3.11 (standardized)

## Migration Guide

### For Users on Python 3.8

1. **Upgrade Python**:
   ```bash
   # Using pyenv
   pyenv install 3.11.7
   pyenv global 3.11.7
   
   # Or using conda
   conda create -n funnel-analytics python=3.11
   conda activate funnel-analytics
   ```

2. **Reinstall Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### For Developers

1. **Update Development Environment**:
   ```bash
   # Ensure you're using Python 3.9+
   python --version
   
   # If not, upgrade your Python installation
   ```

2. **Testing**:
   ```bash
   # All tests should pass on Python 3.9+
   python run_tests.py --all
   ```

## Dependency Versions

Current minimum versions in `requirements.txt`:

```
streamlit>=1.28.0      # Modern UI features
pandas>=2.0.0          # Performance improvements
numpy>=1.24.0          # Vectorized operations
plotly>=5.15.0         # Interactive visualizations
scipy>=1.11.0          # Statistical functions
clickhouse-connect>=0.6.0  # Database connectivity
sqlalchemy>=2.0.0      # ORM features
pyarrow>=12.0.0        # Parquet support
```

## Benefits of Python 3.9+

1. **Performance**: Significant speed improvements in Python 3.9+
2. **Security**: Regular security updates and patches
3. **Features**: Access to latest language features and improvements
4. **Library Support**: Better compatibility with modern libraries
5. **Future-Proofing**: Ensures compatibility with upcoming dependency updates

## Troubleshooting

### Common Issues

1. **Old Python Version**:
   ```
   ERROR: Could not find a version that satisfies the requirement scipy>=1.11.0
   ```
   **Solution**: Upgrade to Python 3.9 or higher

2. **Dependency Conflicts**:
   ```bash
   # Clear pip cache and reinstall
   pip cache purge
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Virtual Environment Issues**:
   ```bash
   # Create fresh virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## Contact

If you encounter issues with Python compatibility, please:
1. Check your Python version: `python --version`
2. Ensure you're using Python 3.9 or higher
3. Create an issue with your environment details 