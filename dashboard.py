import streamlit as st
import pandas as pd
from model_trainer import RolexPriceTrainer
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def load_trainer():
    trainer = RolexPriceTrainer()
    trainer.load_data('Retail_Prices.csv')
    trainer.preprocess_data()
    trainer.split_data()
    return trainer

def train_all_models(trainer):
    models = ["xgboost", "random_forest", "linear", "gradient_boost"]
    results = {}
    
    with st.spinner('Training all models... This may take a few minutes...'):
        progress_bar = st.progress(0)
        for idx, model_type in enumerate(models):
            model_results = trainer.train_model(model_type)
            
            predictions = model_results['predictions']
            y_test = trainer.y_test
            
            rmse = np.sqrt(model_results['mse'])
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            n = len(y_test)
            p = trainer.X_test.shape[1]
            adjusted_r2 = 1 - (1 - model_results['r2']) * (n - 1) / (n - p - 1)
            
            results[model_type] = {
                **model_results,
                'rmse': rmse,
                'mape': mape,
                'adjusted_r2': adjusted_r2
            }
            progress_bar.progress((idx + 1) / len(models))
    return results

def display_model_comparison(results):
    metrics = {
        'mae': {'name': 'Mean Absolute Error', 'format': '${:,.2f}', 'lower_better': True, 'order': 1},
        'mse': {'name': 'Mean Squared Error', 'format': '${:,.2f}', 'lower_better': True, 'order': 2},
        'rmse': {'name': 'Root Mean Squared Error', 'format': '${:,.2f}', 'lower_better': True, 'order': 3},
        'mape': {'name': 'Mean Absolute Percentage Error', 'format': '{:.2f}%', 'lower_better': True, 'order': 4},
        'r2': {'name': 'R² Score', 'format': '{:.3f}', 'lower_better': False, 'order': 5},
        'adjusted_r2': {'name': 'Adjusted R²', 'format': '{:.3f}', 'lower_better': False, 'order': 6}
    }

    table_data = []
    model_scores = {model: 0 for model in results.keys()}

    # Get best values for each metric
    best_values = {}
    for metric in metrics:
        values = [results[model][metric] for model in results.keys()]
        best_values[metric] = min(values) if metrics[metric]['lower_better'] else max(values)

    for model in results.keys():
        row = {'Model': model.upper()}
        for metric, config in metrics.items():
            value = results[model][metric]
            formatted_value = config['format'].format(value)
            
            is_winner = (value == best_values[metric])
            if is_winner:
                formatted_value = f"**{formatted_value}** 🏆"
                model_scores[model] += 1
                
            row[config['name']] = formatted_value
        table_data.append(row)

    df = pd.DataFrame(table_data)
    metric_to_col = {config['name']: config['order'] for metric, config in metrics.items()}
    ordered_cols = ['Model'] + sorted(
        [col for col in df.columns if col != 'Model'],
        key=lambda x: metric_to_col.get(x, 999)
    )
    df = df[ordered_cols]
    
    st.subheader("Model Performance Comparison")
    st.markdown("""
    <style>
    table {
        width: 100%;
        margin-bottom: 1.5em;
    }
    th {
        text-align: left;
        background-color: black;
        padding: 0.5em;
    }
    td {
        text-align: right;
        padding: 0.5em;
    }
    td:first-child {
        text-align: left;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(df.to_markdown(index=False))

    st.subheader("Overall Best Model")
    best_model = max(model_scores.items(), key=lambda x: x[1])
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        ### 🏆 Winner: {best_model[0]}
        Won in {best_model[1]} categories
        
        This model achieved the best performance in {best_model[1]} out of {len(metrics)} metrics.
        """)
    
    with col2:
        scores_data = pd.DataFrame({
            'Model': list(model_scores.keys()),
            'Wins': list(model_scores.values())
        })
        fig = px.bar(scores_data, 
                    x='Wins', 
                    y='Model', 
                    orientation='h',
                    title='Number of Wins by Model')
        fig.update_layout(
            showlegend=False,
            xaxis_title="Number of Metrics Won",
            yaxis_title="Model"
        )
        st.plotly_chart(fig, use_container_width=True)

def create_prediction_interface(trainer):
    valid_values = trainer.get_valid_values()
    
    input_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_data['Reference'] = st.selectbox(
            "Reference",
            options=valid_values['Reference'],
            help="Select the watch reference number"
        )
        
        input_data['Collection'] = st.selectbox(
            "Collection",
            options=valid_values['Collection'],
            help="Select the watch collection"
        )
        
        input_data['Size'] = st.slider(
            "Size (mm)",
            min_value=valid_values['Size']['min'],
            max_value=valid_values['Size']['max'],
            step=valid_values['Size']['step'],
            help="Select the watch size in millimeters"
        )
    
    with col2:
        input_data['Dial'] = st.selectbox(
            "Dial Type",
            options=valid_values['Dial'],
            help="Select the dial type"
        )
        
        bracelet_options = valid_values['Bracelet']
        display_options = [opt if opt != 'None' else 'Standard' for opt in bracelet_options]
        bracelet_index = st.selectbox(
            "Bracelet Type",
            options=display_options,
            help="Select the bracelet type"
        )
        input_data['Bracelet'] = 'None' if bracelet_index == 'Standard' else bracelet_index
        
        input_data['Complication'] = st.multiselect(
            "Complications",
            options=valid_values['Complication'],
            help="Select one or more complications"
        )
    
    return input_data

def main():
    st.set_page_config(page_title="Rolex Price Analysis", layout="wide")
    st.title("Rolex Price Analysis Dashboard")

    if 'trainer' not in st.session_state:
        st.session_state.trainer = load_trainer()
        st.session_state.full_models_trained = False
        
    trainer = st.session_state.trainer
    
    tab1, tab2, tab3 = st.tabs(["Single Model Analysis", "Model Comparison", "Price Predictor"])

    with tab1:
        model_type = st.selectbox(
            "Select Model",
            ["xgboost", "random_forest", "linear", "gradient_boost"]
        )

        with st.spinner('Training model...'):
            results = trainer.train_model(model_type)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"${results['mae']:,.2f}")
        with col2:
            st.metric("Mean Squared Error", f"${results['mse']:,.2f}")
        with col3:
            st.metric("R² Score", f"{results['r2']:.3f}")

        st.subheader("Actual vs Predicted Prices")
        fig = px.scatter(
            x=trainer.y_test, y=results['predictions'],
            labels={'x': 'Actual Price', 'y': 'Predicted Price'}
        )
        fig.add_trace(go.Scatter(
            x=[trainer.y_test.min(), trainer.y_test.max()],
            y=[trainer.y_test.min(), trainer.y_test.max()],
            mode='lines', name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Distribution of Residuals")
        residuals = trainer.y_test - results['predictions']
        fig_residuals = px.histogram(
            residuals, nbins=50,
            labels={'value': 'Residual', 'count': 'Frequency'}
        )
        st.plotly_chart(fig_residuals, use_container_width=True)

    with tab2:
        st.markdown("""
        ### Model Comparison
        Compare all models across different performance metrics to find the best one.
        Each 🏆 indicates the best performing model for that metric.
        """)
        
        if st.button("Compare All Models", type="primary"):
            all_results = train_all_models(trainer)
            display_model_comparison(all_results)
            
        st.info("""
        📊 **Metrics Explained:**
        - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
        - **Mean Squared Error (MSE)**: Average squared difference between predicted and actual prices
        - **Root Mean Squared Error (RMSE)**: Square root of MSE, gives error in same units as price
        - **Mean Absolute Percentage Error (MAPE)**: Average percentage difference from actual price
        - **R² Score**: How well the model explains price variations (higher is better)
        - **Adjusted R²**: R² adjusted for model complexity (higher is better)
        """)

    with tab3:
        st.subheader("Price Predictor")
        
        # Train full models if not already trained
        if not st.session_state.full_models_trained:
            with st.spinner("Training models on full dataset (this will only happen once)..."):
                try:
                    trainer.train_full_models()
                    st.session_state.full_models_trained = True
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    st.stop()
        
        input_data = create_prediction_interface(trainer)
        
        if st.button("Predict Price"):
            if not hasattr(trainer, 'full_models') or not trainer.full_models:
                st.error("Models not trained properly. Please refresh the page.")
            else:
                with st.spinner("Calculating predictions..."):
                    predictions = trainer.predict_price(input_data)
                    
                    if not predictions or predictions.get('mean') is None:
                        st.error("Error calculating predictions. Please check the input values and try again.")
                    else:
                        st.subheader("Model Predictions")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Individual Model Predictions")
                            for model, price in predictions.items():
                                if model != 'mean' and price is not None:
                                    st.metric(
                                        model.upper(), 
                                        f"${price:,.2f}"
                                    )
                        
                        with col2:
                            st.subheader("Average Prediction")
                            st.success(f"""
                            ### 💰 Average Predicted Price
                            
                            **${predictions['mean']:,.2f}**
                            
                            This is the average prediction across all successful model predictions.
                            """)

if __name__ == "__main__":
    main()
