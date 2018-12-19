with
	dates as (
		select calendar_date_start as date_start
		, calendar_date_end as date_end
		, calendar_week_start as week_start
		, to_char(calendar_date_start, 'Day') as day_of_week
		, to_char(date_trunc('month', calendar_date_start), 'Mon') as month
		from public.calendar_dates
		where calendar_date_start >= '1/1/2018'::date
			and calendar_date_start <= getdate()
	)

,	campaign_spend as (
		select "date" as campaign_date
		,	case when region = 'SF' then 'San Francisco Bay Area'
				when region = 'LA' then 'Los Angeles'
				else 'NA' end as region
		, date_trunc('week', "date" - interval '1 week')::date as previous_week_start
		, date_trunc('week', "date")::date as week_start
		, sum(spend) as spend
		from analytics_prod.combined_campaigns
		where region <> 'NA'
			and "date" >= '1/1/2018'::date
		group by 1,2,3,4
	)

, availabilities as (
		select 'Available' as on_website, c.vin,
			convert_timezone('America/Los_angeles', start_at) as start_at_pt,
			convert_timezone('America/Los_angeles', end_at) as end_at_pt,
			c.model_year, c.make, c.model, c.alg_trim,
			c.display_color as color,
			c.region_label as region
		from rome.availability_histories a
		left join rome.cars_clean c on c.car_id = a.car_id
		where 1=1
	)

, reservations as (
		select cs.state_id
		, cs.start_at_pt as reserved_at_pt
		, cs.start_at_pt::date as reserved_date
		, case
		    when extract(hour from cs.start_at_pt) < 10 then '0' + convert(varchar, extract(hour from cs.start_at_pt)) + ':00'
		    else convert(varchar, extract(hour from cs.start_at_pt)) + ':00' end as reserved_hour
		, to_char(cs.start_at_pt, 'HH24:MI') as time_of_day
		, to_char(cs.start_at_pt, 'Day') as day_of_week
		, c.vin, c.model_year, c.make, c.model, c.alg_trim, c.manufacturer_color, c.region_label
		, 1 as is_reserved
		from rome.car_states_clean cs
		left join rome.car_states_clean cs2 on cs2.previous_state_id = cs.state_id
		left join rome.car_states_clean cs3 on cs3.state_id = cs.previous_state_id
		left join rome.car_states_clean cs4 on cs4.previous_state_id = cs2.state_id
		left join rome.cars_clean c on c.car_id = cs.car_id
		where ((cs.category = 'preserved' and cs2.category = 'reserved' and cs4.category = 'contracted')
		  or (cs.category = 'reserved' and cs2.category = 'contracted' and cs3.category <> 'preserved'))
			and cs3.car_state <> 'VehicleHold'
			and cs.start_at_pt::date >= '1/1/2018'::date
	)

, join_tables as (
		select d.*
		, a.region, a.on_website
		, a.vin, a.make, a.model, a.model_year, a.alg_trim, a.color
		, coalesce(r.is_reserved, 0) as is_reserved
-- 		, c.spend as previous_week_spend
		, count(a.vin) over (partition by a.region, date_start) as number_available_cars
		, c.spend as daily_spend
-- 		, c2.spend as current_week_spend
		from dates d
		left join availabilities a on a.start_at_pt < date_end and a.end_at_pt > date_start
		left join reservations r on r.vin = a.vin and r.reserved_date = d.date_start
		left join campaign_spend c on c.campaign_date = d.date_start and c.region = a.region
-- 		left join campaign_spend c2 on c2.week_start = d.week_start and c2.region = a.region
	)

select date_start, region
, extract(day from date_start) as day_num
, to_char(week_start, 'w') as week_num
, month
, coalesce(daily_spend, 0) as daily_spend
, count(distinct vin) as cars_available
, sum(is_reserved) as reservations
, count(distinct case when model = 'Focus' then vin else null end) as focus
, count(distinct case when model = 'Fusion' then vin else null end) as fusion
, count(distinct case when model = 'Escape' then vin else null end) as escape
, count(distinct case when model = 'Explorer' then vin else null end) as explorer
, count(distinct case when model = 'Edge' then vin else null end) as edge
, count(distinct case when model = 'Mustang' then vin else null end) as mustang
, count(distinct case when model = 'C-Max Hybrid' then vin else null end) as cmax_hybrid
, count(distinct case when model = 'Fiesta' then vin else null end) as fiesta
, count(distinct case when model not in ('Focus','Fusion','Escape','Explorer','Edge','Mustang','C-Max Hybrid','Fiesta') then vin else null end) as other
, count(distinct case when model_year = 2015 then vin else null end) as my_2015
, count(distinct case when model_year = 2016 then vin else null end) as my_2016
, count(distinct case when model_year = 2017 then vin else null end) as my_2017
from join_tables
group by 1,2,3,4,5,6
