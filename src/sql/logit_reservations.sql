with
	dates as (
		select calendar_date_start as date_start
		, calendar_date_end as date_end
		, calendar_week_start as week_start
		, to_char(calendar_date_start, 'day') as day_of_week
		, case when to_char(calendar_date_start, 'day') in ('saturday','sunday') then 1 else 0 end as is_weekend
		, to_char(date_trunc('month', calendar_date_start), 'Mon') as month
		from public.calendar_dates
		where calendar_date_start >= '12/14/2017'::date
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
			and "date" >= '12/14/2017'::date
		group by 1,2,3,4
	)

, availabilities as (
		select 'Available' as on_website, c.vin
		,	convert_timezone('America/Los_angeles', start_at) as start_at_pt
		,	convert_timezone('America/Los_angeles', end_at) as end_at_pt
		,	c.model_year, c.make, c.model, c.alg_trim
		, c.trim, c.body_type, c.drivetrain
		,	c.display_color as color
		,	c.region_label as region
		, case when c.trim like 'Hybrid%' then 1 else 0 end as is_hybrid
		, case when c.trim in ('S', 'Hybrid S') then 1 else 0 end as is_s
		, case when c.trim in ('SE', 'Hybrid SE') then 1 else 0 end as is_se
		, case when c.trim in ('Titanium', 'Hybrid Titanium') then 1 else 0 end as is_titanium
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
			and cs.start_at_pt::date >= '12/14/2017'::date
	)

,	vehicle_pricing as (
		select f.fixed_pricing_schedule_id
		, f.id as fixed_pricing_id
		, coalesce(f.total_access_fee, f.vehicle_access_fee)/100 as vehicle_fee
		, model_year || ' ' || make || ' ' || model || ' ' || alg_trim as car_long_name
		, s.version
		, convert_timezone('America/Los_angeles', s.created_at) as pricing_start_pt
		, lead(convert_timezone('America/Los_angeles', s.created_at))
				over (partition by model_year, make, model, alg_trim order by s.created_at) as pricing_end_pt
		from rome.fixed_pricing_items f
		left join rome.fixed_pricing_schedules s on s.id = f.fixed_pricing_schedule_id
		where period = 0
			and s.version is not null
	)

select d.*
, a.region, a.on_website
, a.vin, a.make, a.model, a.model_year, a.alg_trim
, a.body_type, a.trim, a.drivetrain, a.color
, a.is_hybrid, a.is_s, a.is_se, a.is_titanium
, case when d.date_start < '2/28/2018'::date then 0 else 1 end as is_canvas_2_0
, coalesce(r.is_reserved, 0) as is_reserved
, coalesce(c.spend, 0) as daily_spend
, count(a.vin) over (partition by a.region, date_start) as number_available_cars
, v.vehicle_fee
, min(v.vehicle_fee) over (partition by a.region, date_start) as min_vehicle_fee
, avg(v.vehicle_fee) over (partition by a.region, date_start) as avg_vehicle_fee
from dates d
left join availabilities a on a.start_at_pt < date_end and a.end_at_pt > date_start
left join reservations r on r.vin = a.vin and r.reserved_date = d.date_start
left join campaign_spend c on c.campaign_date = d.date_start and c.region = a.region
left join vehicle_pricing v on v.car_long_name = a.model_year || ' ' || a.make || ' ' || a.model || ' ' || a.alg_trim
	and v.pricing_start_pt::date <= d.date_start and v.pricing_end_pt::date > d.date_start
