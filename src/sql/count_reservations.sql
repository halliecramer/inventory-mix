with
	dates as (
		select calendar_date_start as date_start
		, calendar_date_end as date_end
		, calendar_week_start as week_start
		, to_char(calendar_date_start, 'day') as day_of_week
		, case when to_char(calendar_date_start, 'day') in ('saturday', 'sunday') then 1 else 0 end as is_weekend
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
			c.model_year, c.make, c.model, c.alg_trim, c.body_type,
			c.trim, c.drivetrain,
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

, join_tables as (
		select d.*
		, a.region, a.on_website, a.body_type
		, a.vin, a.make, a.model, a.model_year, a.alg_trim, a.color
		, a.trim, a.drivetrain
		, case when d.date_start < '2/28/2018'::date then 0 else 1 end as is_canvas_2_0
		, coalesce(r.is_reserved, 0) as is_reserved
		, count(a.vin) over (partition by a.region, date_start) as number_available_cars
		, c.spend as daily_spend
		, v.vehicle_fee
		from dates d
		left join availabilities a on a.start_at_pt < date_end and a.end_at_pt > date_start
		left join reservations r on r.vin = a.vin and r.reserved_date = d.date_start
		left join campaign_spend c on c.campaign_date = d.date_start and c.region = a.region
		left join vehicle_pricing v on v.car_long_name = a.model_year || ' ' || a.make || ' ' || a.model || ' ' || a.alg_trim
			and v.pricing_start_pt::date <= d.date_start and v.pricing_end_pt::date > d.date_start
	)

select date_start, day_of_week
, replace(lower(region), ' ', '_') as region
, extract(day from date_start) as day_num
, to_char(week_start, 'w') as week_num
, month
, coalesce(daily_spend, 0) as daily_spend
, count(distinct vin) as cars_available
, sum(is_reserved) as reservations
, count(distinct case when body_type = 'sedan' then vin else null end) as sedan
, count(distinct case when body_type = 'suv' then vin else null end) as suv
, count(distinct case when body_type = 'hatchback' then vin else null end) as hatchback
, count(distinct case when body_type = 'wagon' then vin else null end) as wagon
, count(distinct case when body_type in ('convertible','coupe') then vin else null end) as sports_car
, count(distinct case when body_type in ('supercab','supercrew') then vin else null end) as pickup_truck
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
group by 1,2,3,4,5,6,7
