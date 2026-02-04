# # import streamlit as st
# # import pandas as pd
# # import math
# # from pathlib import Path

# # # Set the title and favicon that appear in the Browser's tab bar.
# # st.set_page_config(
# #     page_title='GDP dashboard',
# #     page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
# # )

# # # -----------------------------------------------------------------------------
# # # Declare some useful functions.

# # @st.cache_data
# # def get_gdp_data():
# #     """Grab GDP data from a CSV file.

# #     This uses caching to avoid having to read the file every time. If we were
# #     reading from an HTTP endpoint instead of a file, it's a good idea to set
# #     a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
# #     """

# #     # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
# #     DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
# #     raw_gdp_df = pd.read_csv(DATA_FILENAME)

# #     MIN_YEAR = 1960
# #     MAX_YEAR = 2022

# #     # The data above has columns like:
# #     # - Country Name
# #     # - Country Code
# #     # - [Stuff I don't care about]
# #     # - GDP for 1960
# #     # - GDP for 1961
# #     # - GDP for 1962
# #     # - ...
# #     # - GDP for 2022
# #     #
# #     # ...but I want this instead:
# #     # - Country Name
# #     # - Country Code
# #     # - Year
# #     # - GDP
# #     #
# #     # So let's pivot all those year-columns into two: Year and GDP
# #     gdp_df = raw_gdp_df.melt(
# #         ['Country Code'],
# #         [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
# #         'Year',
# #         'GDP',
# #     )

# #     # Convert years from string to integers
# #     gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

# #     return gdp_df

# # gdp_df = get_gdp_data()

# # # -----------------------------------------------------------------------------
# # # Draw the actual page

# # # Set the title that appears at the top of the page.
# # '''
# # # :earth_americas: GDP dashboard

# # Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
# # notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
# # But it's otherwise a great (and did I mention _free_?) source of data.
# # '''

# # # Add some spacing
# # ''
# # ''

# # min_value = gdp_df['Year'].min()
# # max_value = gdp_df['Year'].max()

# # from_year, to_year = st.slider(
# #     'Which years are you interested in?',
# #     min_value=min_value,
# #     max_value=max_value,
# #     value=[min_value, max_value])

# # countries = gdp_df['Country Code'].unique()

# # if not len(countries):
# #     st.warning("Select at least one country")

# # selected_countries = st.multiselect(
# #     'Which countries would you like to view?',
# #     countries,
# #     ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

# # ''
# # ''
# # ''

# # # Filter the data
# # filtered_gdp_df = gdp_df[
# #     (gdp_df['Country Code'].isin(selected_countries))
# #     & (gdp_df['Year'] <= to_year)
# #     & (from_year <= gdp_df['Year'])
# # ]

# # st.header('GDP over time', divider='gray')

# # ''

# # st.line_chart(
# #     filtered_gdp_df,
# #     x='Year',
# #     y='GDP',
# #     color='Country Code',
# # )

# # ''
# # ''


# # first_year = gdp_df[gdp_df['Year'] == from_year]
# # last_year = gdp_df[gdp_df['Year'] == to_year]

# # st.header(f'GDP in {to_year}', divider='gray')

# # ''

# # cols = st.columns(4)

# # for i, country in enumerate(selected_countries):
# #     col = cols[i % len(cols)]

# #     with col:
# #         first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
# #         last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

# #         if math.isnan(first_gdp):
# #             growth = 'n/a'
# #             delta_color = 'off'
# #         else:
# #             growth = f'{last_gdp / first_gdp:,.2f}x'
# #             delta_color = 'normal'

# #         st.metric(
# #             label=f'{country} GDP',
# #             value=f'{last_gdp:,.0f}B',
# #             delta=growth,
# #             delta_color=delta_color
# #         )

# import streamlit as st
# import pandas as pd
# import math
# import numpy as np
# from pathlib import Path
# import resource
# import os

# # Set the title and favicon
# st.set_page_config(
#     page_title='GDP dashboard',
#     page_icon=':earth_americas:',
# )

# # -----------------------------------------------------------------------------
# # MEMORY INTENSIVE FUNCTIONS

# @st.cache_data
# def get_gdp_data():
#     """Grab GDP data and artificially inflate it to consume memory."""

#     DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
#     raw_gdp_df = pd.read_csv(DATA_FILENAME)

#     # --- MEMORY BLOAT SECTION ---
#     # 1. Duplicate the dataframe rows 2000 times
#     #    If original is ~250 rows, this becomes ~500,000 rows
#     n_copies = 2000
#     raw_gdp_df = pd.concat([raw_gdp_df] * n_copies, ignore_index=True)
    
#     # 2. Add a 'Heavy' column containing long strings (1KB per row)
#     #    500,000 rows * 1KB = ~500MB of raw string data + overhead
#     #    Adjust '1024' to control memory usage. 
#     #    (Note: Pandas overhead often doubles the raw size)
#     raw_gdp_df['memory_bloat'] = 'X' * 1024 
    
#     # ----------------------------

#     MIN_YEAR = 1960
#     MAX_YEAR = 2022

#     # Melt the data (This operation becomes very expensive now)
#     gdp_df = raw_gdp_df.melt(
#         ['Country Code'],
#         [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
#         'Year',
#         'GDP',
#     )

#     # Convert years from string to integers
#     gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

#     return gdp_df

# # Load data (This will now take longer and consume significant RAM)
# with st.spinner('Loading massive dataset to consume memory...'):
#     gdp_df = get_gdp_data()

# # -----------------------------------------------------------------------------
# # Draw the actual page

# '''
# # :earth_americas: GDP dashboard (Memory Intensive Mode)
# '''

# # Display current memory usage (optional debug info)
# # Replacement for psutil memory check
# def get_memory_usage():
#     # usage is in kilobytes on Linux
#     usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#     return usage_kb / 1024  # Convert to MB

# # Add some spacing
# ''
# ''

# min_value = gdp_df['Year'].min()
# max_value = gdp_df['Year'].max()

# from_year, to_year = st.slider(
#     'Which years are you interested in?',
#     min_value=min_value,
#     max_value=max_value,
#     value=[min_value, max_value])

# countries = gdp_df['Country Code'].unique()

# if not len(countries):
#     st.warning("Select at least one country")

# selected_countries = st.multiselect(
#     'Which countries would you like to view?',
#     countries,
#     ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

# ''
# ''
# ''

# # Filter the data
# filtered_gdp_df = gdp_df[
#     (gdp_df['Country Code'].isin(selected_countries))
#     & (gdp_df['Year'] <= to_year)
#     & (from_year <= gdp_df['Year'])
# ]

# st.header('GDP over time', divider='gray')

# ''

# # Aggregate data before plotting because we now have duplicate rows
# # Since we duplicated the data, we must group by Year/Country or the line chart will look messy
# plot_df = filtered_gdp_df.groupby(['Year', 'Country Code'])['GDP'].mean().reset_index()

# st.line_chart(
#     plot_df,
#     x='Year',
#     y='GDP',
#     color='Country Code',
# )

# ''
# ''


# first_year = gdp_df[gdp_df['Year'] == from_year]
# last_year = gdp_df[gdp_df['Year'] == to_year]

# st.header(f'GDP in {to_year}', divider='gray')

# ''

# cols = st.columns(4)

# for i, country in enumerate(selected_countries):
#     col = cols[i % len(cols)]

#     with col:
#         # Adjusted to handle duplicates using .mean() or .iloc[0]
#         first_gdp_data = first_year[first_year['Country Code'] == country]['GDP']
#         last_gdp_data = last_year[last_year['Country Code'] == country]['GDP']

#         if first_gdp_data.empty or last_gdp_data.empty:
#             st.metric(label=f'{country} GDP', value="No Data")
#             continue

#         first_gdp = first_gdp_data.iloc[0] / 1000000000
#         last_gdp = last_gdp_data.iloc[0] / 1000000000

#         if math.isnan(first_gdp):
#             growth = 'n/a'
#             delta_color = 'off'
#         else:
#             growth = f'{last_gdp / first_gdp:,.2f}x'
#             delta_color = 'normal'

#         st.metric(
#             label=f'{country} GDP',
#             value=f'{last_gdp:,.0f}B',
#             delta=growth,
#             delta_color=delta_color
#         # )



import streamlit as st
import pandas as pd
import numpy as np
import math
import resource  # Standard lib alternative to psutil
from pathlib import Path

# Set page config
st.set_page_config(
    page_title='GDP High Load',
    page_icon=':warning:',
)

# -----------------------------------------------------------------------------
# MEMORY & CPU STRESS FUNCTIONS

def get_memory_usage():
    """Returns current process memory usage in MB using standard library."""
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024

@st.cache_data
def load_and_bloat_data():
    """
    Loads data and creates massive in-memory objects to stress RAM.
    """
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    # --- STAGE 1: DATAFRAME EXPLOSION (Text Data) ---
    # Duplicate rows 5000 times (approx 1.2 million rows)
    # This creates a very "tall" dataframe
    n_copies = 5000
    st.toast(f"Allocating {n_copies} copies of dataframe...")
    big_df = pd.concat([raw_gdp_df] * n_copies, ignore_index=True)

    # --- STAGE 2: HEAVY MATRIX ALLOCATION (Numerical Data) ---
    # Create a massive random float matrix (approx 800MB - 1GB raw RAM)
    # Shape: 10,000 rows x 10,000 cols
    st.toast("Generating massive 10k x 10k matrix...")
    heavy_matrix = np.random.rand(10000, 10000) 
    
    # Force a calculation to ensure memory is actually committed (COW handling)
    # This matrix multiplication is CPU and RAM heavy
    temp_result = np.sum(heavy_matrix) 
    
    # Store this heavy object in the dataframe to keep it in memory
    # We just store the sum reference or a slice to prove we did it, 
    # but keeping 'heavy_matrix' in this scope keeps it alive if cached.
    
    return big_df, heavy_matrix.nbytes / (1024**2)

# -----------------------------------------------------------------------------
# APP LOGIC

st.title(':warning: High-Load GDP Dashboard')
st.caption("Stress testing memory limits with massive objects.")

# Metrics container
metric_col1, metric_col2 = st.columns(2)

with st.spinner('Occupying Memory... (This may take a moment)'):
    try:
        gdp_df, matrix_size_mb = load_and_bloat_data()
        
        # Current Mem Usage
        current_mem = get_memory_usage()
        
        metric_col1.metric("Current Memory Usage", f"{current_mem:.0f} MB")
        metric_col2.metric("Matrix Object Size", f"{matrix_size_mb:.0f} MB")
        
        if current_mem > 2500:
            st.error(f"⚠️ CRITICAL MEMORY WARNING: {current_mem:.0f} MB used")
        else:
            st.success(f"System Stable: {current_mem:.0f} MB used")
            
    except Exception as e:
        st.error(f"Crashed due to OOM or Error: {e}")
        st.stop()

# -----------------------------------------------------------------------------
# DATA PROCESSING (The heavy dataframe)

MIN_YEAR = 1960
MAX_YEAR = 2022

# Melt the massive dataframe (Expensive operation)
# We limit to first 100k rows for the UI to remain responsive, 
# even though the full object sits in RAM.
display_df = gdp_df.iloc[:100000].melt(
    ['Country Code'],
    [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
    'Year',
    'GDP',
)

display_df['Year'] = pd.to_numeric(display_df['Year'])

# -----------------------------------------------------------------------------
# VISUALIZATION

countries = display_df['Country Code'].unique()
selected_countries = st.multiselect(
    'Filter Countries (Data sampled from first 100k rows)',
    countries,
    ['DEU', 'FRA', 'GBR'])

if selected_countries:
    filtered_df = display_df[display_df['Country Code'].isin(selected_countries)]
    
    # Group by because we have duplicates from the explosion
    chart_data = filtered_df.groupby(['Year', 'Country Code'])['GDP'].mean().reset_index()
    
    st.line_chart(chart_data, x='Year', y='GDP', color='Country Code')

# Add a button to intentionally spike memory further
if st.button("💣 SPIKE MEMORY (+500MB)"):
    # This creates a new list of 500MB random data that persists 
    # as long as the script runs this rerun
    spike = np.random.bytes(500 * 1024 * 1024)
    st.warning(f"Allocated additional 500MB. New Usage: {get_memory_usage():.0f} MB")
    st.write(f"Spike object length: {len(spike)}")
    