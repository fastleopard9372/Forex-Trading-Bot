def run_backtest(self, start_date, end_date):
    """
    Run backtest over a specific period
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        
    Returns:
        dict: Backtest results
    """
    if not self.initialize_mt5():
        logger.error("Failed to initialize MT5. Exiting.")
        return None
    
    logger.info(f"Starting backtest from {start_date} to {end_date}")
    
    try:
        results = {}
        
        for symbol in self.symbols:
            logger.info(f"Backtesting {symbol}...")
            
            # Fetch historical data for all timeframes
            df_long = self.fetch_historical_data(symbol, self.timeframes['long_term'], start_date, end_date)
            df_medium = self.fetch_historical_data(symbol, self.timeframes['medium_term'], start_date, end_date)
            df_short = self.fetch_historical_data(symbol, self.timeframes['short_term'], start_date, end_date)
            df_execution = self.fetch_historical_data(symbol, self.timeframes['execution'], start_date, end_date)
            
            if df_long is None or df_medium is None or df_short is None or df_execution is None:
                logger.error(f"Failed to fetch data for {symbol}. Skipping.")
                continue
            
            # Calculate indicators
            df_long = self.calculate_indicators(df_long)
            df_medium = self.calculate_indicators(df_medium)
            df_short = self.calculate_indicators(df_short)
            df_execution = self.calculate_indicators(df_execution)
            
            # Reset account balance for each symbol
            current_balance = self.initial_balance
            equity_curve = [current_balance]
            trades = []
            active_trades = []
            
            # Define mapping between execution timeframe and analysis timeframes
            # This helps align indices when trading on lower timeframes
            timeframe_map = {}
            
            # Map execution timeframe to other timeframes using timestamps
            for exec_idx, exec_time in enumerate(df_execution.index):
                timeframe_map[exec_idx] = {
                    'long': df_long.index.get_indexer([exec_time], method='pad')[0],
                    'medium': df_medium.index.get_indexer([exec_time], method='pad')[0],
                    'short': df_short.index.get_indexer([exec_time], method='pad')[0]
                }
            
            # Track market conditions and strategies used
            market_conditions_log = []
            strategies_used = []
            
            # Iterate through each execution bar (2-minute timeframe)
            # Skip the first set of bars to ensure indicators are calculated
            start_idx = max(self.slow_period, self.rsi_period) + 50  # Add margin for stability
            
            for idx in range(start_idx, len(df_execution)):
                current_time = df_execution.index[idx]
                
                # Get mapped indices
                mapped_indices = timeframe_map.get(idx, {
                    'long': 0,
                    'medium': 0,
                    'short': 0
                })
                
                idx_long = mapped_indices['long']
                idx_medium = mapped_indices['medium']
                idx_short = mapped_indices['short']
                
                # Check for any closed trades
                for trade in active_trades[:]:
                    updated_trade, profit = self.update_trade_status(
                        trade, 
                        df_execution.iloc[idx], 
                        current_time,
                        df_short,
                        idx_short
                    )
                    
                    if updated_trade['status'] != 'open':
                        active_trades.remove(trade)
                        trades.append(updated_trade)
                        current_balance += profit
                        equity_curve.append(current_balance)
                
                # Check for new trade signals if we don't have an active trade for this symbol
                if not any(t['symbol'] == symbol for t in active_trades):
                    # Analyze market condition
                    market_condition = self.analyze_market_condition(
                        df_long, df_medium, df_short, 
                        idx_long, idx_medium, idx_short
                    )
                    
                    # Log market condition
                    market_conditions_log.append({
                        'time': current_time,
                        'condition': market_condition
                    })
                    
                    # Determine if we should trade
                    should_trade, order_type, strategy_name = self.determine_trade_strategy(
                        market_condition, symbol, df_short, idx_short
                    )
                    
                    if should_trade:
                        # Log strategy
                        strategies_used.append({
                            'time': current_time,
                            'strategy': strategy_name,
                            'order_type': order_type
                        })
                        
                        # Determine optimal trade parameters
                        sl_pips, tp_pips, trailing_stop = self.determine_trade_parameters(
                            market_condition, symbol, df_execution, idx
                        )
                        
                        # Calculate position size
                        lot_size = self.calculate_position_size(symbol, sl_pips, current_balance)
                        
                        # Get entry price
                        entry_price = df_execution['close'].iloc[idx]
                        
                        # Simulate trade
                        trade = self.simulate_trade(
                            symbol=symbol,
                            order_type=order_type,
                            entry_price=entry_price,
                            sl_pips=sl_pips,
                            tp_pips=tp_pips,
                            trailing_stop=trailing_stop,
                            lot_size=lot_size,
                            entry_time=current_time,
                            strategy=strategy_name
                        )
                        
                        active_trades.append(trade)
                        logger.info(f"New trade: {symbol} {order_type} at {entry_price}, " 
                                    f"SL: {sl_pips} pips, TP: {tp_pips} pips, " 
                                    f"Strategy: {strategy_name}")
            
            # Close any remaining open trades at the last price
            for trade in active_trades:
                trade['exit_price'] = df_execution['close'].iloc[-1]
                trade['exit_time'] = df_execution.index[-1]
                trade['status'] = 'closed'
                trade['exit_reason'] = 'end_of_test'
                
                # Calculate profit
                if trade['type'] == "BUY":
                    profit_pips = (trade['exit_price'] - trade['entry_price']) / (mt5.symbol_info(trade['symbol']).point * 10)
                else:  # SELL
                    profit_pips = (trade['entry_price'] - trade['exit_price']) / (mt5.symbol_info(trade['symbol']).point * 10)
                
                if 'JPY' in trade['symbol']:
                    profit_pips = profit_pips * 10  # Adjust for JPY pairs
                
                trade['profit_pips'] = profit_pips
                
                # Calculate profit amount
                symbol_info = mt5.symbol_info(trade['symbol'])
                pip_value = symbol_info.trade_contract_size * symbol_info.point
                profit_amount = profit_pips * pip_value * trade['lot_size']
                trade['profit_amount'] = profit_amount
                
                current_balance += profit_amount
                equity_curve.append(current_balance)
                trades.append(trade)
            
            # Analyze strategies performance
            strategy_performance = {}
            for trade in trades:
                strategy = trade['strategy']
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_profit': 0,
                        'total_pips': 0
                    }
                
                strategy_performance[strategy]['total_trades'] += 1
                strategy_performance[strategy]['total_profit'] += trade['profit_amount']
                strategy_performance[strategy]['total_pips'] += trade['profit_pips']
                
                if trade['profit_amount'] > 0:
                    strategy_performance[strategy]['winning_trades'] += 1
            
            # Calculate win rates for strategies
            for strategy, perf in strategy_performance.items():
                if perf['total_trades'] > 0:
                    perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
                    perf['avg_profit_per_trade'] = perf['total_profit'] / perf['total_trades']
                    perf['avg_pips_per_trade'] = perf['total_pips'] / perf['total_trades']
                else:
                    perf['win_rate'] = 0
                    perf['avg_profit_per_trade'] = 0
                    perf['avg_pips_per_trade'] = 0
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t['profit_amount'] > 0])
            losing_trades = len([t for t in trades if t['profit_amount'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_profit = sum([t['profit_amount'] for t in trades])
            average_profit = total_profit / total_trades if total_trades > 0 else 0
            
            winning_amounts = [t['profit_amount'] for t in trades if t['profit_amount'] > 0]
            losing_amounts = [abs(t['profit_amount']) for t in trades if t['profit_amount'] <= 0]
            
            average_win = sum(winning_amounts) / len(winning_amounts) if winning_amounts else 0
            average_loss = sum(losing_amounts) / len(losing_amounts) if losing_amounts else 0
            
            profit_factor = sum(winning_amounts) / sum(losing_amounts) if sum(losing_amounts) > 0 else float('inf')
            
            # Calculate drawdown
            peak = self.initial_balance
            drawdown = 0
            max_drawdown = 0
            drawdown_periods = []
            
            for i, balance in enumerate(equity_curve):
                if balance > peak:
                    peak = balance
                
                current_drawdown = (peak - balance) / peak * 100 if peak > 0 else 0
                drawdown = current_drawdown
                
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
                    
                # Track drawdown periods
                if i > 0:
                    prev_drawdown = (peak - equity_curve[i-1]) / peak * 100 if peak > 0 else 0
                    if prev_drawdown == 0 and current_drawdown > 0:
                        # Start of drawdown period
                        drawdown_periods.append({
                            'start_idx': i,
                            'start_balance': equity_curve[i-1],
                            'peak': peak
                        })
                    elif prev_drawdown > 0 and current_drawdown == 0:
                        # End of drawdown period
                        if drawdown_periods and 'end_idx' not in drawdown_periods[-1]:
                            drawdown_periods[-1]['end_idx'] = i
                            drawdown_periods[-1]['end_balance'] = balance
                            drawdown_periods[-1]['drawdown_pct'] = (drawdown_periods[-1]['peak'] - 
                                                                    min(equity_curve[drawdown_periods[-1]['start_idx']:i+1])) / drawdown_periods[-1]['peak'] * 100
                            drawdown_periods[-1]['duration'] = i - drawdown_periods[-1]['start_idx']
            
            # Calculate market condition statistics
            market_condition_stats = {
                'strong_uptrend': 0,
                'strong_downtrend': 0,
                'sideways': 0,
                'high_volatility': 0
            }
            
            for record in market_conditions_log:
                condition = record['condition']
                if condition['strong_uptrend']:
                    market_condition_stats['strong_uptrend'] += 1
                if condition['strong_downtrend']:
                    market_condition_stats['strong_downtrend'] += 1
                if condition['sideways']:
                    market_condition_stats['sideways'] += 1
                if condition['high_volatility']:
                    market_condition_stats['high_volatility'] += 1
            
            # Calculate percentage of time in each condition
            total_bars = len(market_conditions_log)
            if total_bars > 0:
                for key in market_condition_stats:
                    market_condition_stats[f"{key}_pct"] = market_condition_stats[key] / total_bars * 100
            
            # Store results
            results[symbol] = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': average_profit,
                'average_win': average_win,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'drawdown_periods': drawdown_periods,
                'final_balance': current_balance,
                'equity_curve': equity_curve,
                'trades': trades,
                'strategy_performance': strategy_performance,
                'market_condition_stats': market_condition_stats
            }
            
            logger.info(f"Backtest for {symbol} completed. Win rate: {win_rate:.2%}, Profit: {total_profit:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"An error occurred during backtest: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    finally:
        # Shutdown MT5
        mt5.shutdown()
        logger.info("MT5 connection closed")

def plot_results(self, results):
    """
    Plot backtest results
    
    Args:
        results (dict): Backtest results
    """
    if not results:
        logger.error("No results to plot")
        return
    
    # Create figure and subplots
    plt.figure(figsize=(16, 24))
    
    # Determine number of symbols
    num_symbols = len(results)
    subplot_count = 0
    
    # 1. Plot equity curves
    subplot_count += 1
    plt.subplot(num_symbols + 5, 1, subplot_count)
    plt.title("Equity Curves", fontsize=14)
    plt.xlabel("Trades")
    plt.ylabel("Balance")
    
    for symbol, result in results.items():
        plt.plot(result['equity_curve'], label=symbol)
    
    plt.legend()
    plt.grid(True)
    
    # 2. Plot drawdown
    subplot_count += 1
    plt.subplot(num_symbols + 5, 1, subplot_count)
    plt.title("Drawdown", fontsize=14)
    
    for symbol, result in results.items():
        equity = np.array(result['equity_curve'])
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        plt.plot(drawdown, label=f"{symbol} (Max: {result['max_drawdown']:.2f}%)")
    
    plt.legend()
    plt.grid(True)
    plt.ylabel("Drawdown (%)")
    
    # 3. Strategy comparison
    subplot_count += 1
    plt.subplot(num_symbols + 5, 1, subplot_count)
    plt.title("Strategy Performance", fontsize=14)
    
    # Collect all strategies
    all_strategies = set()
    for result in results.values():
        all_strategies.update(result['strategy_performance'].keys())
    
    # Prepare data for bar chart
    strategies = list(all_strategies)
    win_rates = []
    avg_profits = []
    
    for strategy in strategies:
        strategy_win_rates = []
        strategy_avg_profits = []
        
        for result in results.values():
            if strategy in result['strategy_performance']:
                perf = result['strategy_performance'][strategy]
                strategy_win_rates.append(perf['win_rate'])
                strategy_avg_profits.append(perf['avg_profit_per_trade'])
        
        win_rates.append(np.mean(strategy_win_rates) if strategy_win_rates else 0)
        avg_profits.append(np.mean(strategy_avg_profits) if strategy_avg_profits else 0)
    
    # Plot win rates
    x = np.arange(len(strategies))
    width = 0.35
    plt.bar(x - width/2, [w * 100 for w in win_rates], width, label='Win Rate (%)')
    plt.bar(x + width/2, avg_profits, width, label='Avg Profit per Trade')
    
    plt.xlabel('Strategy')
    plt.xticks(x, strategies, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    
    # 4. Market Condition Distribution
    subplot_count += 1
    plt.subplot(num_symbols + 5, 1, subplot_count)
    plt.title("Market Condition Distribution", fontsize=14)
    
    condition_labels = ['Strong Uptrend', 'Strong Downtrend', 'Sideways', 'High Volatility']
    condition_keys = ['strong_uptrend_pct', 'strong_downtrend_pct', 'sideways_pct', 'high_volatility_pct']
    
    for symbol, result in results.items():
        condition_pcts = [result['market_condition_stats'].get(key, 0) for key in condition_keys]
        plt.bar(np.arange(len(condition_labels)) + (list(results.keys()).index(symbol) * 0.2), 
                condition_pcts, width=0.2, label=symbol)
    
    plt.xlabel('Market Condition')
    plt.ylabel('Percentage of Time (%)')
    plt.xticks(np.arange(len(condition_labels)), condition_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    
    # 5. Individual symbol results
    for i, (symbol, result) in enumerate(results.items(), 1):
        subplot_count += 1
        plt.subplot(num_symbols + 5, 1, subplot_count)
        plt.title(f"{symbol} - Win Rate: {result['win_rate']:.2%}, Profit: {result['total_profit']:.2f}", fontsize=14)
        
        # Extract trade data
        trade_numbers = list(range(1, len(result['trades']) + 1))
        profits = [t['profit_amount'] for t in result['trades']]
        strategies = [t['strategy'] for t in result['trades']]
        
        # Create unique color map for strategies
        unique_strategies = list(set(strategies))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_strategies)))
        strategy_colors = {s: colors[i] for i, s in enumerate(unique_strategies)}
        
        # Create bar colors based on strategy
        bar_colors = [strategy_colors[s] for s in strategies]
        
        plt.bar(trade_numbers, profits, color=bar_colors)
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel("Trade Number")
        plt.ylabel("Profit/Loss")
        
        # Create legend for strategies
        legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=strategy) 
                            for strategy, color in strategy_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("backtest_results.png")
    plt.close()

def print_summary(self, results):
    """
    Print a summary of backtest results
    
    Args:
        results (dict): Backtest results
    """
    if not results:
        logger.error("No results to summarize")
        return
    
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    
    for symbol, result in results.items():
        print(f"\nSymbol: {symbol}")
        print(f"Total Trades: {result['total_trades']}")
        print(f"Win Rate: {result['win_rate']:.2%}")
        print(f"Total Profit: {result['total_profit']:.2f}")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Average Win: {result['average_win']:.2f}")
        print(f"Average Loss: {result['average_loss']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
        print(f"Final Balance: {result['final_balance']:.2f}")
        
        print("\nStrategy Performance:")
        for strategy, perf in result['strategy_performance'].items():
            if perf['total_trades'] > 0:
                print(f"  {strategy}:")
                print(f"    Trades: {perf['total_trades']}")
                print(f"    Win Rate: {perf['win_rate']:.2%}")
                print(f"    Avg Profit: {perf['avg_profit_per_trade']:.2f}")
                print(f"    Avg Pips: {perf['avg_pips_per_trade']:.2f}")
        
        print("\nMarket Conditions:")
        for condition, pct in result['market_condition_stats'].items():
            if "_pct" in condition:
                print(f"  {condition.replace('_pct', '')}: {pct:.2f}%")
        
        print("-"*80)
    
    # Calculate combined results
    total_trades = sum([r['total_trades'] for r in results.values()])
    winning_trades = sum([r['winning_trades'] for r in results.values()])
    total_profit = sum([r['total_profit'] for r in results.values()])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Find max drawdown across all symbols
    max_drawdown = max([r['max_drawdown'] for r in results.values()])
    
    # Calculate combined strategy performance
    combined_strategy_perf = {}
    
    for result in results.values():
        for strategy, perf in result['strategy_performance'].items():
            if strategy not in combined_strategy_perf:
                combined_strategy_perf[strategy] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'total_profit': 0,
                    'total_pips': 0
                }
            
            combined_strategy_perf[strategy]['total_trades'] += perf['total_trades']
            combined_strategy_perf[strategy]['winning_trades'] += perf['winning_trades']
            combined_strategy_perf[strategy]['total_profit'] += perf['total_profit']
            combined_strategy_perf[strategy]['total_pips'] += perf['total_pips']
    
    # Calculate win rates
    for strategy, perf in combined_strategy_perf.items():
        if perf['total_trades'] > 0:
            perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
            perf['avg_profit'] = perf['total_profit'] / perf['total_trades']
            perf['avg_pips'] = perf['total_pips'] / perf['total_trades']
        else:
            perf['win_rate'] = 0
            perf['avg_profit'] = 0
            perf['avg_pips'] = 0
    
    print("\nCOMBINED RESULTS:")
    print(f"Total Trades: {total_trades}")
    print(f"Overall Win Rate: {win_rate:.2%}")
    print(f"Total Profit: {total_profit:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Final Account Balance: {self.initial_balance + total_profit:.2f}")
    
    print("\nCombined Strategy Performance:")
    for strategy, perf in combined_strategy_perf.items():
        if perf['total_trades'] > 0:
            print(f"  {strategy}:")
            print(f"    Trades: {perf['total_trades']}")
            print(f"    Win Rate: {perf['win_rate']:.2%}")
            print(f"    Avg Profit: {perf['avg_profit']:.2f}")
            print(f"    Avg Pips: {perf['avg_pips']:.2f}")
    
    print("="*80)


if __name__ == "__main__":
    # Create and run the backtest
    backtest = TradingBotBacktest(
        symbols=["USDJPY", "EURUSD"],
        risk_percent=0.02,  # Risk 2% per trade
        sl_pips_default=30,  # 30 pips default stop loss
        tp_multiplier_default=2,  # Take profit is 2x stop loss by default
        fast_period=12,
        slow_period=26,
        signal_period=9,
        rsi_period=14,
        rsi_overbought=70,
        rsi_oversold=30,
        initial_balance=10000
    )

    # Define backtest period
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Run backtest
    results = backtest.run_backtest(start_date, end_date)

    # Print summary
    backtest.print_summary(results)

    # Plot results
    backtest.plot_results(results)