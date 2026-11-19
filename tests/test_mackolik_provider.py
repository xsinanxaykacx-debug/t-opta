from mackolik_provider import parse_match_odds


def test_parse_current_match_result_odds_from_mackolik_html():
    html = """
    <main>
      <ul>
        <li>
          <h2>Maç Sonucu MBS 3</h2>
          <ul>
            <li><span>1</span><span>2.68</span></li>
            <li><span>X</span><span>2.34</span></li>
            <li><span>2</span><span>2.30</span></li>
          </ul>
          <ul>
            <li><span>1</span><span>2.88</span></li>
            <li><span>X</span><span>2.54</span></li>
            <li><span>2</span><span>2.50</span></li>
          </ul>
        </li>
      </ul>
    </main>
    """
    assert parse_match_odds(html) == (2.68, 2.34, 2.30)


def test_parse_match_result_odds_with_comma_decimal_separator():
    html = """
    <li>
      <h2>Maç Sonucu</h2>
      <ul>
        <li>1 1,75</li>
        <li>X 3,20</li>
        <li>2 4,10</li>
      </ul>
    </li>
    """
    assert parse_match_odds(html) == (1.75, 3.20, 4.10)


def test_missing_match_result_market_returns_none():
    assert parse_match_odds("<main><h2>Çifte Şans</h2></main>") is None
