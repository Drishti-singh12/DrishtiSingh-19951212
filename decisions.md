decisions.md — Analytical Decision Log
A chronological record of decisions made, dead ends hit, and things that surprised me during this assessment.

Before Opening Any File — Hypotheses
Before touching the data I wrote three hypotheses as instructed:

Cancellation contacts will spike within 24–48 hours of booking, driven by price-regret behaviour — customers second-guessing a purchase shortly after making it.
Phone channel will carry disproportionate contact volume for high-value orders — customers spending more money are more anxious and less willing to wait for a chat reply.
Repeat contacts are concentrated in a small subset of orders — I expected a 80/20 pattern where roughly 20% of contacted orders drive 80% of total contact volume.

I'll revisit these at the end. Spoiler: I was wrong on hypothesis 1 in a way I didn't expect.

Hour 1 — The Join Key Problem
The first thing I did was load both files and check shapes. Orders had 6.3 million rows. Errands had 2.88 million. Straightforward enough. Then I looked at the join key.
order_id in the orders table is a plain integer — something like 4607745402. order_number in the errands table looks completely different — something like 24770FC. My first assumption was that these were two separate internal identifiers that would require a lookup table I hadn't been given. I nearly stopped and drafted a note saying "linkage not possible without additional reference data."
Then I looked more carefully. The order_number values are always 7 characters. They contain digits and uppercase letters up to F. That looked like hexadecimal. I tried converting order_id to hex — no match. Then I tried base-36, which uses 0–9 and A–Z. 24770FC in base-36 is 4607513832. That's in the same range as the order_id integers.
I tested it on five rows. All five matched. Ran the full join. 100% match rate on real errands.
This was the hidden anomaly mentioned in the brief. If I had joined on the raw strings, or assumed the fields were already compatible, I would have silently lost every single errand-to-order link, and every downstream analysis would have been based on nothing.
Decision: convert order_number to integer via base-36 before any join. Documented in code comments.

Hour 1.5 — Test Errand Filtering
The is_test_errand flag was easy to spot but easy to forget. 204,404 records — 7.1% of raw errands — are test records. I filtered these immediately after loading and before computing any rates.
What caught my attention: the test records are not evenly distributed. They cluster in specific time windows, which suggests batch testing events rather than ongoing background noise. I decided not to investigate further — it's operationally interesting but not what the VP of CS asked about. Flagged it in the quality audit and moved on.

Hour 2 — The Contact Rate Surprise
I expected a contact rate somewhere around 25–35%. The actual number is 16.3%. My first reaction was to question whether the join had worked properly. I double-checked the base-36 conversion, re-ran the match, confirmed 100% linkage. The number is just 16.3%.
This is actually meaningful. It means that despite being a high-volume CS environment, most bookings complete without any contact at all. The question becomes: what makes the 16.3% different? That framed the entire statistical section.

Hour 2.5 — Hypothesis 1 Was Wrong
When I ran the time-to-first-contact analysis I expected contacts to cluster in the first 24–48 hours — the "price regret" window. The median time to first contact is 16 days. Only 23% of first contacts happen within 24 hours of booking.
This completely reframed my understanding of the data. Customers are not contacting CS because they immediately regret their purchase. They are contacting CS because something goes wrong later — a schedule change, a cancellation by the airline, a document question before travel. The contact driver is disruption, not regret.
This made the root cause chart for the executive deck obvious: show the contact rate for normal orders vs changed orders vs cancelled orders. The gap (14% → 60% → 71%) is the story.
I abandoned the "price regret" narrative entirely. In the executive deck there is no mention of it. The decisions diary is where I'm being honest that I walked in with the wrong hypothesis.

Hour 3 — Statistical Test Selection
For the propensity factors I had five tests to design. I considered three options for the categorical variables (Is_Canceled, Is_Changed, Customer_Group_Type, Journey_Type_ID): chi-square, Fisher's exact, or logistic regression coefficient tests.
Fisher's exact is theoretically preferred for small samples but becomes computationally impractical with 500,000 rows and multi-category variables. Chi-square is appropriate when expected cell counts are all > 5, which is trivially satisfied at this sample size.
For Order_Amount (continuous) against the binary contact flag, I chose Mann-Whitney U rather than a t-test. The Order_Amount distribution is extremely right-skewed (median €15, mean €2,507 — an enormous gap caused by a small number of very high-value bookings). A t-test assumes roughly normal distributions. Mann-Whitney makes no such assumption and tests whether the distributions are stochastically different. It's the right choice here.
One thing I was careful about: every p-value in this dataset is going to be < 0.001 simply because of sample size. With 500,000 observations, even trivially small differences achieve statistical significance. I reported effect sizes alongside p-values everywhere, and made the distinction explicit. A chi-square of 45,000 with p=0 and a chi-square of 64 with p=2e-10 are both "significant" but they represent very different practical realities.

Hour 4 — Machine Learning Decisions
Feature leakage was my biggest concern here. The target variable is "did this order generate a CS contact?". Many columns in the orders table are filled in after the booking takes place — Is_Canceled, Is_Changed, cancel_reason, change_reason. Including any of these as model features would be catastrophic leakage: the model would be predicting from the future.
I excluded all post-booking columns from the feature set. The remaining features are: brand, booking system, customer group type, device, acquisition channel, booking system source type, journey type, currency, and time-of-booking features (hour, day of week, month).
The resulting AUC is 0.606. That's honest. It's not impressive. But I think it's a genuinely useful number because it tells us something important: at the time of booking, we cannot reliably predict contact propensity. The factors that drive contacts — airline schedule changes, weather events, pre-travel anxiety in the final 48 hours — simply don't exist yet. The model is best used to route high-risk segments to premium support channels at booking time, not to forecast contact volume.
I compared Logistic Regression and Random Forest. RF outperforms on both AUC (0.606 vs 0.586) and average precision. The top feature in both models by a large margin is log(Order_Amount), which aligns with the statistical finding that high-value orders contact at twice the rate of low-value ones.
Decision: recommend RF for deployment, but only for channel routing at high confidence thresholds. Be explicit in the executive deck that this is a guidance tool, not a volume forecast.

Hour 4.5 — Clustering Choices
For segmentation I initially tried k=2 to k=6 with silhouette scoring. k=2 had the best silhouette (0.38) but produces only "contacted" vs "not contacted" — not actionable. I chose k=4 because the four segments map cleanly to a business taxonomy: cancelled orders, modified orders, round-trip safe bookers, and simple one-way travellers.
The silhouette scores for k=3 through k=6 were all in the 0.11–0.18 range, which is modest. I considered DBSCAN as an alternative since it doesn't require specifying k. The challenge is that with 80,000 rows and 10 features, DBSCAN is sensitive to epsilon and runs slowly. Given the clear business rationale for 4 clusters, I stayed with KMeans and was transparent about the modest silhouette.

Biggest Surprises

The join key was base-36. I've never encountered this in a real dataset before. If I hadn't stopped and thought carefully about the character set, I would have concluded the datasets couldn't be joined.
Most contacts are not immediate. The 16-day median TTFC genuinely surprised me. It forces a completely different story about what CS is actually handling.
Schedule change contacts have a 70% repeat rate — higher than cancellation contacts. Intuitively I assumed cancellations (the most emotionally charged event) would drive the most repeat behaviour. The data says otherwise. Customers who first contact about a schedule change are more likely to call back again than customers who first contact about a cancellation. This might be because cancellations have a clear resolution path (refund), while schedule changes require ongoing negotiation.


What I Would Do With More Time

Normalise all Order_Amount figures to a single currency (EUR) using daily FX rates. The current analysis treats a £500 booking and a €500 booking as equivalent, which they're not.
Build a cost model by estimating per-contact costs per channel. Without cost data, the opportunity sizing remains directional rather than precise.
Run the survival analysis properly using the Kaplan-Meier estimator with the lifelines library (unavailable in the analysis environment, so I implemented the empirical survival function manually).
Investigate the temporal decline in contact volume after July 2024 more deeply — is it a seasonal booking pattern, a genuine service improvement, or a data completeness issue in recent months?