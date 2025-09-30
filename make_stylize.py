import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# ======= Set current folder as root =======
# ==========================================
script_dir = pathlib.Path(__file__).resolve().parent
os.chdir(script_dir)

# Use LaTeX-style font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# ==========================================
# ========== Figure 1(a) ===================
# ==========================================
data = pd.read_excel('japan_stylized_data.xlsx', sheet_name='data_1a')
df = pd.DataFrame(data)

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.bar(df['year'], df['male_abs'], color='lightblue', label='male')
ax1.bar(df['year'], df['female_abs'], bottom=df['male_abs'], color='lightcoral', label='female')
ax1.set_xlabel('year', size=20)
ax1.set_ylabel('thousands people', size=20)
ax1.set_xticks(df['year'][::4].dropna().astype(int))
ax1.legend(fontsize=18)
fig1.tight_layout()
fig1.savefig('fig_leaving_for_kaigo.pdf', format='pdf', bbox_inches='tight', dpi=300)


# ==========================================
# ========== Figure 1(b) ===================
# ==========================================
data2 = pd.read_excel('japan_stylized_data.xlsx', sheet_name='data_1b')
df2 = pd.DataFrame(data2[:-1])

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(df2['year'], df2['partcipants_regular_total'], label='regular (full-time)', marker='o')
ax2.plot(df2['year'], df2['participants_parttime_total'], label='part-time', marker='s')
ax2.set_xlabel('year', size=20)
ax2.set_ylabel('weekly hours', size=20)
ax2.set_xticks(np.arange(2001, 2017, 5))
ax2.legend(loc='upper left', fontsize=18)
fig2.tight_layout()
fig2.savefig('fig_weekly_timeuse_kaigo.pdf', format='pdf', bbox_inches='tight', dpi=300)


# ==========================================
# ========== Figure 2(a) ===================
# ==========================================
sib_data = pd.read_excel('japan_stylized_data.xlsx', sheet_name='data_2a')
sib_df = pd.DataFrame(sib_data).drop([0])

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(sib_df['birth_year'], sib_df['nshc03t_table27'], label='Third Survey')
ax3.plot(sib_df['birth_year'], sib_df['nshc04t'], label='Fourth Survey')
ax3.plot(sib_df['birth_year'], sib_df['nshc05t'], label='Fifth Survey')
ax3.plot(sib_df['birth_year'], sib_df['nshc06t'], label='Sixth Survey')
ax3.plot(sib_df['birth_year'], sib_df['nshc07t'], label='Seventh Survey')
ax3.plot(sib_df['birth_year'], sib_df['nshc08t'], label='Eighth Survey')
ax3.set_xlabel("Cohort's Birth Year", fontsize=15)
ax3.set_ylabel('Average Number of Siblings', fontsize=15)
ax3.legend(fontsize=18)
ax3.tick_params(axis='x', labelrotation=90, labelsize=20)
ax3.tick_params(axis='y', labelsize=20)
fig3.tight_layout()
fig3.savefig('fig_jp_siblings_survey.pdf', format='pdf', bbox_inches='tight', dpi=300)


# ==========================================
# ========== Figure 2(b) ===================
# ==========================================
care_data = pd.read_excel('japan_stylized_data.xlsx', sheet_name='data_2b')
care_df = pd.DataFrame(care_data)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.plot(care_df['year'], care_df['child'], label="children and children's spouses", marker='o')
ax4.plot(care_df['year'], care_df['care_support_provider'], label='formal care', marker='s')
ax4.set_xlabel('Year', fontsize=20)
ax4.set_ylabel(r'Percentage (\%) of elderly requiring care', fontsize=20)
ax4.set_xticks(np.arange(2001, 2020, 3))
ax4.legend(fontsize=18)
fig4.tight_layout()
fig4.savefig('fig_jp_principal_care_update.pdf', format='pdf', bbox_inches='tight', dpi=300)